import tensorflow as tf

import numpy as np
import json
from input import DataLoader
from rason_model import biGRU_model

NUM_STEPS = 40
VOCAB_SIZE = 30000
BATCH_SIZE = 10
MAX_EPOCH = 10

def train(model, dataLoader):
    global_step = 0
    
    for i in range(MAX_EPOCH):
    
        cost_per_epoch = 0.
        batch_count = 0
        batch_generator = dataLoader.batch_gen(BATCH_SIZE)
        
        for batch in batch_generator:
            
            batch_count += 1
            global_step += 1
                
            cost = model.train(batch)
            print ('global_step', str(global_step), ',Cost of Epoch', str(i), 'batch', str(batch_count), ":", str(cost))
            
            cost_per_epoch += cost
            
            #if global_step%200 == 0:
                #test(model,test_datas,batch_size,max_sentence_len,word_dimension,global_step)

        print ('Cost of Epoch', str(i), ":", str( cost_per_epoch / batch_count))
        
def main():

    with open('data/train_set', 'r') as f:
        word_sequence = json.loads(f.read())
        
    word_sequence = [word for sentence in word_sequence for word in sentence]
    
    dataLoader = DataLoader(word_sequence, NUM_STEPS, VOCAB_SIZE)
    
    model = biGRU_model(batch_size = BATCH_SIZE, 
                        num_steps = NUM_STEPS, 
                        vocab_size = VOCAB_SIZE,
                        num_hidden = 800,
                        dropout_rate = 0.5,
                        num_layers = 2)
    
    model.initialize()
    
    train(model, dataLoader) 
            
if __name__ == "__main__":
    main()
