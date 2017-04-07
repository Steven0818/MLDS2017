import sys
import json
import numpy as np

import util
import input

VOCAB_SIZE = 3000
FRAME_STEP = 80
FRAME_DIM = 4096
BATCH_SIZE = 20
CAPTION_STEP = 30

train_npy_path = 'data/training_data/feat'

def main():

    #build_word2idx_dict(vocab_size=VOCAB_SIZE,
    #                    trainlable_json='data/training_label.json', 
    #                    testlabel_json='data/testing_public_label.json',
    #                    dict_path='data/dict.json')

    d_word2idx = json.load(open('data/dict.json', 'r'))
    tr_in_idx = get_tr_in_idx(trainlable_json='data/training_label.json', dict_path='data/dict.json')
    
    dataLoader = input.DataLoader(tr_in_idx,
                                  data_path='data/training_data/feat', 
                                  frame_step=FRAME_STEP,
                                  frame_dim=FRAME_DIM,
                                  caption_step=CAPTION_STEP
                                 )
    
    batch_generator = dataLoader.batch_gen(BATCH_SIZE)
    
    #for x, y in batch_generator:


if __name__ == '__main__':
    main()