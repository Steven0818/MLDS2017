import sys
import json
import numpy as np
import model
import util
import input
import eval

VOCAB_SIZE = 3000
FRAME_STEP = 80
FRAME_DIM = 4096
BATCH_SIZE = 20
CAPTION_STEP = 45
EPOCH = 1000

train_npy_path = 'data/training_data/feat'

def trim(sen):
    if 3 in sen:
        return sen[:sen.index(3)]
    else:
        return sen

def test(model, test_data, dict_rev, global_step, output_path='result/'):
    answers = []
    # print(sum(1 for i in test_batch_generator))
    score = 0
    for x, video_ids, captions in test_data:
        result = model.predict(x)
        ## remove eos
        sentences = [' '.join([dict_rev[str(word)] for word in trim(sen.tolist())]) for sen in result[0]]
        answers.extend(list(zip(video_ids, sentences)))
        for index, answer in enumerate(sentences):
            score += np.mean(np.array([eval.BLEU(answer, cap) for cap in captions[index]]))
    print('BLEU score of step {0}: {1}'.format(global_step, score/len(answers)))
    json.dump([{'caption': cap, 'id:': vid} for vid, cap in answers],
              open(output_path + 'result_' +str(global_step) + '.json', 'w'))
        


def main():

    # util.build_word2idx_dict(vocab_size=VOCAB_SIZE,
    #                    trainlable_json='data/training_label.json', 
    #                    testlabel_json='data/testing_public_label.json',
    #                    dict_path='data/dict.json')

    print ("building model...")
    S2VT = model.S2VT_model(caption_steps=CAPTION_STEP)
    S2VT.initialize()
    print ("building model successfully...")
    
    d_word2idx = json.load(open('data/dict.json', 'r'))
    d_idx2word = json.load(open('data/dict_rev.json', 'r'))
    tr_in_idx = util.get_tr_in_idx(trainlable_json='data/training_label.json', dict_path='data/dict.json')
    test_label = json.load(open('data/testing_public_label.json'))

    dataLoader = input.DataLoader(tr_in_idx,
                                  data_path='data/training_data/feat',
                                  frame_step=FRAME_STEP,
                                  frame_dim=FRAME_DIM,
                                  caption_step=CAPTION_STEP,
                                  vocab_size=VOCAB_SIZE
                                 )
    test_data_loader = input.TestDataLoader(test_label,
                                        data_path='data/testing_data/feat',
                                        frame_step=FRAME_STEP,
                                        frame_dim=FRAME_DIM,
                                        caption_step=CAPTION_STEP,
                                        vocab_size=VOCAB_SIZE,
                                        shuffle=False
                                        )
    test_batch = test_data_loader.get_data(BATCH_SIZE)

    global_step = 0
    
    print ("training start....")
    for i in range(EPOCH):
        batch_generator = dataLoader.batch_gen(BATCH_SIZE)
        batch_count = 0
        for x, y , y_mask in batch_generator:
            cost = S2VT.train(x,y,y_mask)
            global_step += 1
            batch_count += 1
            if global_step % 100 == 0:
                print('global_step {0} cost: {1}'.format(global_step, cost))
            if global_step % 1000 == 0:
                test(S2VT, test_batch, d_idx2word, global_step)
        print('Epoch {0} end:'.format(i))

if __name__ == '__main__':
    main()
