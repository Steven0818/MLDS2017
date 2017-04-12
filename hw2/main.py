import sys
import json
import numpy as np
import model
import util
import input

VOCAB_SIZE = 3000
FRAME_STEP = 80
FRAME_DIM = 4096
BATCH_SIZE = 20
CAPTION_STEP = 45

train_npy_path = 'data/training_data/feat'

def main():

    # util.build_word2idx_dict(vocab_size=VOCAB_SIZE,
    #                    trainlable_json='data/training_label.json', 
    #                    testlabel_json='data/testing_public_label.json',
    #                    dict_path='data/dict.json')

    print ("building model...")
    S2VT = model.S2VT_attention_model(batch_size=BATCH_SIZE, caption_steps=CAPTION_STEP)
    S2VT.initialize()
    print ("building model successfully...")
    
    d_word2idx = json.load(open('data/dict.json', 'r'))
    tr_in_idx = util.get_tr_in_idx(trainlable_json='data/training_label.json', dict_path='data/dict.json')



    dataLoader = input.DataLoader(tr_in_idx,
                                  data_path='data/training_data/feat', 
                                  frame_step=FRAME_STEP,
                                  frame_dim=FRAME_DIM,
                                  caption_step=CAPTION_STEP,
                                  vocab_size=VOCAB_SIZE
                                 )
    
    batch_generator = dataLoader.batch_gen(BATCH_SIZE)
        
    print ("training start....")
    for x, y , y_mask in batch_generator:
        cost = S2VT.train(x,y,y_mask)

if __name__ == '__main__':
    main()
