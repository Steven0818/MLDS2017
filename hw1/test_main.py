import tensorflow as tf
import json
import numpy as np
import csv
import sys
from rason_model import biGRU_model


NUM_STEPS = 40
VOCAB_SIZE = 30000
EMBEDDING_DIM = 300
BATCH_SIZE = 20
MAX_EPOCH = 10
NUM_HIDDEN = 800
KEEP_PROB = 0.5

def main():
        
    with open('test.json', 'r') as f:
        test_data = json.loads(f.read())
    
    model = biGRU_model(num_steps = NUM_STEPS,
                        vocab_size = VOCAB_SIZE,
                        num_hidden = NUM_HIDDEN,
                        num_layers = 2)
    
    model.loadModel('./model_50000.ckpt')
    
    test(model, test_data, 0)
    
def test(model, test_data, global_step):    
    answer_list = ["a","b","c","d","e"]
    
    predict_list = []
    predict_list.append(["id","answer"])
    
    input_data = []
    answer_word_index = []
    answer_option = []
    question_count = 0
    
    for data in test_data:
        
        for index,value in enumerate(data["sentence"]):
            if value == -1:
                answer_word_index.append(index)
        
        input_data.append(data["sentence"])
        answer_option.append(data["answer"])
        
        if len(input_data) % BATCH_SIZE == 0:
            input_sentences = index2vector(input_data, NUM_STEPS, VOCAB_SIZE)

            ### (batch_size,(max_sentence_len-2),word_dimension)
            predict_result = model.predict(input_sentences, keep_prob=1.)  
            
            for i, predict_distribute in enumerate(predict_result):
                #answer_word_index[i] - 1 for the predict_result starts from the 2nd word in sentence
                answer_prob_distribute = np.asarray(predict_distribute[answer_word_index[i] - 1])[answer_option[i]]
                question_count += 1
                predict_pair = [question_count, answer_list[np.argmax(answer_prob_distribute)]]
                predict_list.append(predict_pair)
            
            answer_word_index = []
            answer_option = []
            input_data = []

    if len(input_data) != 0:
        input_sentences = index2vector(input_data, NUM_STEPS, VOCAB_SIZE)
        ### (batch_size,(max_sentence_len-2),word_dimension)
        predict_result = model.predict(input_sentences)  
            
        for i, predict_distribute in enumerate(predict_result):
            answer_prob_distribute = np.asarray(predict_distribute[answer_word_index[i] - 1])[answer_option[i]]
            question_count += 1
            predict_pair = [question_count,answer_list[np.argmax(answer_prob_distribute)]]
            predict_list.append(predict_pair)

    with open(sys.argv[1], "w") as f:  
        w = csv.writer(f)  
        w.writerows(predict_list)
    
    print ("finish prediction.....")

def index2vector(input_data, num_steps, vocab_size):
    
    sentences = np.zeros([len(input_data), num_steps, vocab_size])
    
    for i,data in enumerate(input_data):
        for j,one_hot_index in enumerate(data):
            sentences[i,j,one_hot_index] = 1

    return sentences    
    
if __name__ == "__main__":
    main()