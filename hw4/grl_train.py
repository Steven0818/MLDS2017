from __future__ import division
from __future__ import print_function


import sys
import os
import time
import math
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import gst_rnn_model
import grl_rnn_model
import data_utils
import conf
import pickle
import os.path

gst_config = conf.GSTConfig
gbk_config = conf.GBKConfig
grl_config = conf.GRLConfig
gcc_config = conf.GCCConfig
pre_grl_config = conf.Pre_GRLConfig

def read_data(config, source_path, target_path, max_size=None):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.strip().split()]
                target_ids = [int(x) for x in target.strip().split()]
                # source_ids.insert(0,data_utils._GO)
                # source_ids.append(data_utils.EOS_ID)
                # target_ids.append(data_utils._GO))
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def prepare_data(config):
    train_path = os.path.join(config.train_dir, "movie_subtitle.train")
    data_path_list = [train_path + ".answer", train_path + ".query"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    data_utils.create_vocabulary(vocab_path, data_path_list, config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    #
    # if os.path.isfile(config.dev_set) and os.path.isfile(config.train_set):
    #     dev_set_file = open(config.dev_set, "rb")
    #     dev_set = pickle.load(dev_set_file)
    #     dev_set_file.close()
    #
    #     train_set_file = open(config.train_set, "rb")
    #     train_set = pickle.load(train_set_file)
    #     train_set_file.close()
    # else:
    print("Prepare Movie Subtitle data in %s" % config.train_dir)
    train_query, train_answer = data_utils.prepare_chitchat_data(
        config.train_dir, vocab, config.vocab_size)

    print("Reading development and training data (limit: %d)." % config.max_train_data_size)
    # dev_set = read_data(config, dev_query, dev_answer)
    train_set = read_data(config, train_query, train_answer)

        # dev_set_file = open(config.dev_set, "wb")
        # pickle.dump(dev_set, dev_set_file)
        # dev_set_file.close()
        #
        # train_set_file = open(config.train_set, "wb")
        # pickle.dump(train_set, train_set_file)
        # train_set_file.close()

    return vocab, rev_vocab, train_set


def create_st_model(session, st_config, forward_only, name_scope):
    with tf.variable_scope(name_or_scope=name_scope):
        st_model = gst_rnn_model.gst_model(gst_config=st_config, name_scope=name_scope, forward_only=forward_only)
        ckpt = tf.train.get_checkpoint_state(os.path.join(st_config.train_dir, "checkpoints"))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Read %s model from %s" % (name_scope, ckpt.model_checkpoint_path))
            st_model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Creating %s model with fresh parameters" % name_scope)
            global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            session.run(tf.variables_initializer(global_variables))
            print("Created %s model with fresh parameters" % name_scope)
        return st_model


def create_rl_model(session, rl_config, forward_only, name_scope, dummy_set):
    with tf.variable_scope(name_or_scope=name_scope):
        rl_model = grl_rnn_model.grl_model(grl_config=rl_config, name_scope=name_scope, forward=forward_only,dummy_set=dummy_set)
        ckpt = tf.train.get_checkpoint_state(os.path.join(rl_config.train_dir, "checkpoints"))
        print (ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Read %s model from %s" % (name_scope, ckpt.model_checkpoint_path))
            rl_model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Creating %s model with fresh parameters" % name_scope)
            global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            session.run(tf.variables_initializer(global_variables))
            print("Created %s model with fresh parameters" % name_scope)
        return rl_model


def ce_standard_train(st_config):
    vocab, rev_vocab, train_set = prepare_data(st_config)
    for b_set in train_set:
        print("b_set length: ", len(b_set))

    with tf.Session() as sess:
        print("Creating %s %d layers of %d units" %(st_config.name_model ,st_config.num_layers, st_config.emb_dim))
        st_model = create_st_model(sess, st_config, False, st_config.name_model)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(st_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        step_loss_summary = tf.Summary()
        # merge = tf.merge_all_summaries()
        st_writer = tf.summary.FileWriter(st_config.tensorboard_dir, sess.graph)

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            #print("bucket_id: ", bucket_id)
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = \
                st_model.get_batch(train_set, bucket_id)
            _, step_loss, _ = st_model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                            forward_only=False)

            step_time += (time.time() - start_time) / st_config.steps_per_checkpoint
            loss += step_loss / st_config.steps_per_checkpoint
            current_step += 1
            
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % st_config.steps_per_checkpoint == 0:
                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = st_config.name_loss
                bucket_value.simple_value = float(loss)
                st_writer.add_summary(step_loss_summary, int(sess.run(st_model.global_step)))

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (st_model.global_step.eval(), st_model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(st_model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                gen_ckpt_dir = os.path.abspath(os.path.join(st_config.train_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                    os.makedirs(gen_ckpt_dir)
                checkpoint_path = os.path.join(gen_ckpt_dir, "movie_subtitle.model")
                st_model.saver.save(sess, checkpoint_path, global_step=st_model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def pre_rl_train(rl_config):
    vocab, rev_vocab, train_set = prepare_data(rl_config)
    for b_set in train_set:
        print("b_set length: ", len(b_set))

    with tf.Session() as sess:
        rl_model = create_rl_model(sess, rl_config=rl_config, forward_only=False, name_scope=rl_config.name_model)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(rl_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        step_loss_summary = tf.Summary()
        rl_writer = tf.summary.FileWriter(rl_config.tensorboard_dir, sess.graph)

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, _ = \
                rl_model.get_batch(train_set,bucket_id)

            _, _, step_loss = rl_model.step(sess, encoder_inputs, decoder_inputs,target_weights,
                                       reward=1, bucket_id=bucket_id, forward_only=False, beam_search=False)

            step_time += (time.time() - start_time) / rl_config.steps_per_checkpoint
            loss += step_loss / rl_config.steps_per_checkpoint
            current_step += 1

            if current_step % rl_config.steps_per_checkpoint == 0:

                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = rl_config.pre_name_loss
                bucket_value.simple_value = float(loss)
                rl_writer.add_summary(step_loss_summary, int(sess.run(rl_model.global_step)))

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (rl_model.global_step.eval(), rl_model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(rl_model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                gen_ckpt_dir = os.path.abspath(os.path.join(rl_config.train_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                    os.makedirs(gen_ckpt_dir)
                checkpoint_path = os.path.join(gen_ckpt_dir, "movie_subtitle.model")
                rl_model.saver.save(sess, checkpoint_path, global_step=rl_model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()
    pass


def train():
    vocab, rev_vocab, train_set = prepare_data(grl_config)
    for b_set in train_set:
        print("b_set length: ", len(b_set))
    dummy_set = data_utils.get_dummy_set("grl_data/dummy_sentence",vocab,25000)
    with tf.Session() as sess:
        rl_model = create_rl_model(sess, grl_config, False, grl_config.name_model,dummy_set)
        st_model = create_st_model(sess, gst_config, True, gst_config.name_model)
        #bk_model = create_st_model(sess, gbk_config, True, gbk_config.name_model)
        #cc_model = create_st_model(sess, gcc_config, True, gcc_config.name_model)
        

        train_bucket_sizes = [len(train_set[b]) for b in range(len(grl_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        step_loss_summary = tf.Summary()
        # merge = tf.merge_all_summaries()
        rl_writer = tf.summary.FileWriter(grl_config.tensorboard_dir, sess.graph)
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, _ = \
                rl_model.get_batch(train_set,bucket_id)
           
            updata, norm, step_loss = rl_model.step_rl(sess, st_model=st_model, bk_model=st_model, cc_model = st_model,encoder_inputs=encoder_inputs,
                                               decoder_inputs=decoder_inputs, target_weights=target_weights,
                                               batch_source_encoder=batch_source_encoder, bucket_id=bucket_id)

            step_time += (time.time() - start_time) / grl_config.steps_per_checkpoint
            loss += step_loss / grl_config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % grl_config.steps_per_checkpoint == 0:

                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = grl_config.name_loss
                bucket_value.simple_value = float(loss)
                rl_writer.add_summary(step_loss_summary, int(sess.run(rl_model.global_step)))

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (rl_model.global_step.eval(), rl_model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(rl_model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                gen_ckpt_dir = os.path.abspath(os.path.join(grl_config.train_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                    os.makedirs(gen_ckpt_dir)
                checkpoint_path = os.path.join(gen_ckpt_dir, "movie_subtitle.model")
                rl_model.saver.save(sess, checkpoint_path, global_step=rl_model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def test_decoder(config):
    train_path = os.path.join(config.train_dir, "movie_subtitle.train")
    data_path_list = [train_path + ".answer", train_path + ".query"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
#    data_utils.create_vocabulary(vocab_path, data_path_list, config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path) 
    dummy_set = data_utils.get_dummy_set("grl_data/dummy_sentence",vocab,25000)
    with tf.Session() as sess:
        if config.name_model in [gst_config.name_model, gcc_config.name_model, gbk_config.name_model]:
            model = create_st_model(sess, config, forward_only=True, name_scope=config.name_model)

        elif config.name_model in [grl_config.name_model, pre_grl_config.name_model]:
            model = create_rl_model(sess, config, forward_only=True, name_scope=config.name_model,dummy_set = dummy_set)

        model.batch_size = 1

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
            print("token_id: ", token_ids)
            bucket_id = len(config.buckets) - 1
            for i, bucket in enumerate(config.buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                print("Sentence truncated: %s", sentence)

            encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch({bucket_id: [(token_ids, [1])]},
                                                                                   bucket_id)
            # st_model step
            if config.name_model in [gst_config.name_model, gcc_config.name_model, gbk_config.name_model]:
                output_logits, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                print(" ".join([str(rev_vocab[output]) for output in outputs]))

            # beam_search step
            elif config.name_model in [grl_config.name_model, pre_grl_config.name_model]:
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, reward=1,
                                                 bucket_id=bucket_id, forward_only=True)
                #output_logits = np.reshape(output_logits,[1,-1,25000])
                output_logits = np.squeeze(output_logits)
                outputs = np.argmax(output_logits, axis=1)
                outputs = list(outputs)
                # for i, output in enumerate(output_logits):
                #     print("index: %d, answer tokens: %s" %(i, str(output)))
                #     if data_utils.EOS_ID in output:
                #         output = output[:output.index(data_utils.EOS_ID)]
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                print (outputs)
                while data_utils.UNK_ID in outputs:
                    sub_max = np.argmax(output_logits[outputs.index(data_utils.UNK_ID)][4:])+4
                    outputs[outputs.index(data_utils.UNK_ID)] = sub_max
                print(" ".join([str(rev_vocab[out]) for out in outputs]))


            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
def read_file_test(config,test_model_name,input_path,output_path):
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path) 
    dummy_set = data_utils.get_dummy_set("grl_data/dummy_sentence",vocab,25000)
    forward_only = True
    with tf.Session() as sess:
        with tf.variable_scope(name_or_scope=config.name_model):
            model = grl_rnn_model.grl_model(grl_config=config, name_scope=config.name_model, forward=forward_only,dummy_set=dummy_set)
        #ckpt = tf.train.get_checkpoint_state(os.path.join(rl_config.train_dir, "checkpoints"))
        #print (ckpt.model_checkpoint_path)
            model.batch_size = 1
            if test_model_name == 'S2S':
                model.saver.restore(sess,"grl_data/movie_subtitle.model-118000")
            elif test_model_name == 'RL':
                model.saver.restore(sess,"grl_data/movie_subtitle.model-127200")
            else:
                model.saver.restore(sess,"grl_data/movie_subtitle.model-127200")
            with open(input_path) as f:
                sentences = f.readlines()
            output_file = []
            for sentence in sentences:    
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
                print("token_id: ", token_ids)
                bucket_id = len(config.buckets) - 1
                for i, bucket in enumerate(config.buckets):
                    if bucket[0] >= len(token_ids):
                        bucket_id = i
                        break
                else:
                    print("Sentence truncated: %s", sentence)

                encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch({bucket_id: [(token_ids, [1])]},
                                                                                   bucket_id)
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, reward=1,
                                                 bucket_id=bucket_id, forward_only=True)
                #output_logits = np.reshape(output_logits,[1,-1,25000])
                output_logits = np.squeeze(output_logits)
                outputs = np.argmax(output_logits, axis=1)
                outputs = list(outputs)
                
                # for i, output in enumerate(output_logits):
                #     print("index: %d, answer tokens: %s" %(i, str(output)))
                #     if data_utils.EOS_ID in output:
                #         output = output[:output.index(data_utils.EOS_ID)]
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                print (outputs)
                while data_utils.UNK_ID in outputs:
                    sub_max = np.argmax(output_logits[outputs.index(data_utils.UNK_ID)][4:])+4
                    outputs[outputs.index(data_utils.UNK_ID)] = sub_max
                while 30 in outputs: outputs.remove(30)

                output_sentence = " ".join([str(rev_vocab[out]) for out in outputs])+'\n'
                #while '$' in output_sentence:
                output_sentence = output_sentence.replace("$", "")
                print (output_sentence)
                output_file.append(output_sentence)
            f = open(output_path, 'w')
            f.writelines(output_file)

def decoder(config):
    vocab, rev_vocab, train_set = prepare_data(config)

    train_bucket_sizes = [len(train_set[b]) for b in range(len(config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    with tf.Session() as sess:
        model = create_st_model(sess, config, forward_only=True, name_scope=config.name_model)

        disc_train_query = open("train.query", "w")
        disc_train_answer = open("train.answer", "w")
        disc_train_gen = open("train.gen", "w")

        num_step = 0
        while num_step < 50000:
            print("generating num_step: ", num_step)
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = \
                model.get_batch(train_set, bucket_id)

            out_logits, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

            tokens = []
            resps = []
            for seq in out_logits:
                # print("seq: %s" %seq)
                token = []
                for t in seq:
                    # print("seq_t: %s" %t)
                    # t = list(t)
                    # print("list(t): %s" %t)
                    # t = np.array(t)
                    # print("array(t): %s" %t)
                    token.append(int(np.argmax(t, axis=0)))
                tokens.append(token)
            tokens_t = []
            for col in range(len(tokens[0])):
                tokens_t.append([tokens[row][col] for row in range(len(tokens))])

            for seq in tokens_t:
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID)][:config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:config.buckets[bucket_id][1]])

            for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):

                answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
                disc_train_answer.write(answer_str)
                disc_train_answer.write("\n")

                query_str = " ".join([str(rev_vocab[qu]) for qu in query])
                disc_train_query.write(query_str)
                disc_train_query.write("\n")

                resp_str = " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])

                disc_train_gen.write(resp_str)
                disc_train_gen.write("\n")
            num_step += 1

        disc_train_gen.close()
        disc_train_query.close()
        disc_train_answer.close()
    pass


def main(_):
    # model_1 P_backward(qi|a)
    # ce_standard_train(gbk_config)

    # model_2 P(a|pi,qi)
    # ce_standard_train(gcc_config)

    #model_3 P(s|a)
    #ce_standard_train(gst_config)

    # model_4.1 pre P_rl
    #pre_rl_train(pre_grl_config)

    # model_4.2 P_rl
    # train()

    read_file_test(grl_config,sys.argv[1],sys.argv[2],sys.argv[3])
    #test_decoder(grl_config)


if __name__ == "__main__":
    tf.app.run()
