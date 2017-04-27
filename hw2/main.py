import sys
import json
import numpy as np
import model
import util
import input
import eval
import time

VOCAB_SIZE = 3000
FRAME_STEP = 20
FRAME_DIM = 4096
BATCH_SIZE = 100
CAPTION_STEP = 45
EPOCH = 1000
SCHEDULED_SAMPLING_CONVERGE = 5000
MODEL_FILE_NAME = 'result_schedule'


train_npy_path = 'data/training_data/feat'

def trim(sen):
    if 3 in sen:
        return sen[:sen.index(3)]
    else:
        return sen

def test(model, test_data, dict_rev, global_step, output_path= MODEL_FILE_NAME+'/' , train_test='test'):
    answers = []
    score = 0
    for x, video_ids, captions in test_data:
        result = model.predict(x)
        ## remove eos
        sentences = [' '.join([dict_rev[str(word)] for word in trim(sen.tolist())]) for sen in result[0]]
        answers.extend(list(zip(video_ids, sentences)))
        for index, answer in enumerate(sentences):
            score += np.mean(np.array([eval.BLEU(answer, cap) for cap in captions[index]]))
    print('{0} BLEU score of step {1}: {2}'.format(train_test, global_step, score/len(answers)))
    json.dump([{'caption': cap, 'id:': vid} for vid, cap in answers],
              open(output_path + train_test + '_result_' +str(global_step) + '.json', 'w'))
        


def main():


    print ("building model...")
    S2VT = model.Effective_attention_model(caption_steps=CAPTION_STEP)
    # S2VT.loadModel('result_schedule_para/model_10000.ckpt')
    # print ("loading sucessfully")
    # S2VT.initialize()
    # print ("building model successfully...")
    
    # d_word2idx = json.load(open('data/dict.json', 'r'))
    # d_idx2word = json.load(open('data/dict_rev.json', 'r'))
    # tr_in_idx = util.get_tr_in_idx(trainlable_json='data/training_label.json', dict_path='data/dict.json')
    # test_label = json.load(open('data/testing_public_label.json'))

    # # dataLoader = input.DataLoader(tr_in_idx,
    # #                               data_path='data/training_data/feat',
    # #                               frame_step=FRAME_STEP,
    # #                               frame_dim=FRAME_DIM,
    # #                               caption_step=CAPTION_STEP,
    # #                               vocab_size=VOCAB_SIZE
    # #                              )
    # data = util.Data(
    #     'data/training_data/feat',
    #     json.load(open('data/training_label.json')),
    #     d_word2idx,
    #     d_idx2word,
    #     BATCH_SIZE)
    # test_data_loader = input.TestDataLoader(test_label,
    #                                     data_path='data/testing_data/feat',
    #                                     frame_step=FRAME_STEP,
    #                                     frame_dim=FRAME_DIM,
    #                                     caption_step=CAPTION_STEP,
    #                                     vocab_size=VOCAB_SIZE,
    #                                     shuffle=False
    #                                     )
    # train_label = json.load(open('data/training_label.json'))
    # train_test_data_loader = input.TestDataLoader(train_label,
    #                                               data_path = 'data/training_data/feat',
    #                                               frame_step = FRAME_STEP,
    #                                               frame_dim = FRAME_DIM,
    #                                               caption_step=CAPTION_STEP,
    #                                               vocab_size=VOCAB_SIZE,
    #                                               shuffle=False)
    # test_batch = test_data_loader.get_data(BATCH_SIZE)
    # train_test_batch = train_test_data_loader.get_data(BATCH_SIZE)

    # global_step = 0
    # epoch_count = 1
    # epoch_size = len(data)
    # print (epoch_size)
    # loader = data.loader()
    # print ("training start....")
    # for i in range(int(len(data) * EPOCH / BATCH_SIZE)):  
        
    #     frames, captions, target_weights = next(loader)
    #     start_time = time.time()
    #     cost = S2VT.train(
    #         frames, captions, target_weights, scheduled_sampling_prob=i / SCHEDULED_SAMPLING_CONVERGE)
    #     global_step += 1
    #     finish_time = time.time()
    #     #print ('each step time cost: {0}'.format(finish_time-start_time))
    #     if global_step % 100 == 0:
    #         print('global_step {0} cost: {1}'.format(global_step, cost))
    #     if global_step % 2000 == 0:
    #         test(S2VT, test_batch, d_idx2word, global_step,train_test = 'test')
    #         test(S2VT, train_test_batch, d_idx2word, global_step,train_test = 'train')
    #         S2VT.saveModel(MODEL_FILE_NAME)
    #     if i * BATCH_SIZE > epoch_count * epoch_size:
    #         print('Epoch {0} end'.format(epoch_count))
    #         epoch_count += 1


if __name__ == '__main__':
    main()
