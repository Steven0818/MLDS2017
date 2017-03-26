import tensorflow as tf

import numpy as np
import json
from input import DataLoader2
from rason_model import biGRU_model, biGRU_nce_model

NUM_STEPS = 40
VOCAB_SIZE = 30000
BATCH_SIZE = 20
MAX_EPOCH = 10
num_hidden = 600

def train(model, dataLoader):
    global_step = 0
    
    for i in range(MAX_EPOCH):
    
        cost_per_epoch = 0.
        batch_count = 0
        batch_generator= dataLoader.batch_gen(BATCH_SIZE)
        
        for x in batch_generator:
            
            batch_count += 1
            global_step += 1
                
            cost = model.train(x)
            print ('global_step', str(global_step), ',Cost of Epoch', str(i), 'batch', str(batch_count), ":", str(cost))
            
            cost_per_epoch += cost
            
            #if global_step%200 == 0:
                #test(model,test_datas,batch_size,max_sentence_len,word_dimension,global_step)

        print ('Cost of Epoch', str(i), ":", str( cost_per_epoch / batch_count))

def test(model, test_data, global_step):    
    answer_list = ["a","b","c","d","e"]
    
    predict_list = []
	predict_list.append(["id","answer"])
	
    input_data = []
	answer_word_index = []
	answer_option = []
	question_count = 0
    
	for data in test_datas:
		for index,value in enumerate(data["sentence"]):
			if value == -1:
				answer_word_index.append(index)
		input_data.append(data["sentence"])
		answer_option.append(data["answer"])
		if len(input_data)%batch_size == 0:
			input_sentences = index2vector(input_data,max_sentence_len,word_dimension)

			### (batch_size,(max_sentence_len-2),word_dimension)
			predict_result = model.predict(input_sentences)  
			
			for i,predict_distribute in enumerate(predict_result):
				answer_prob_distribute = np.asarray(predict_distribute[answer_word_index[i]-1])[answer_option[i]]
				question_count+=1
				predict_pair = [question_count,answer_list[np.argmax(answer_prob_distribute)]]
				predict_list.append(predict_pair)
			answer_word_index = []
			answer_option = []
			input_data = []

	if len(input_data)!=0:
		input_sentences = index2vector(input_data,max_sentence_len,word_dimension)
		### (batch_size,(max_sentence_len-2),word_dimension)
		predict_result = model.predict(input_sentences)  
			
		for i,predict_distribute in enumerate(predict_result):
			answer_prob_distribute = np.asarray(predict_distribute[answer_word_index[i]-1])[answer_option[i]]
			question_count+=1
			predict_pair = [question_count,answer_list[np.argmax(answer_prob_distribute)]]
			predict_list.append(predict_pair)


	f = open("csv/test_"+str(global_step)+".csv","w")  
	w = csv.writer(f)  
	w.writerows(predict_list)  
	print ("finish prediction.....")

        
def main():

    with open('data/train_set', 'r') as f:
        word_sequence = json.loads(f.read())
        
    word_sequence = [word for sentence in word_sequence for word in sentence]
    
    dataLoader = DataLoader2(word_sequence, NUM_STEPS, VOCAB_SIZE)
    batch_count_epoch = dataLoader.batch_count(BATCH_SIZE)
    
    model = biGRU_nce_model(num_steps = NUM_STEPS, 
                        vocab_size = VOCAB_SIZE,
                        num_hidden = num_hidden,
                        dropout_rate = 0.5,
                        num_layers = 2)
    
    model.initialize()
    
    print("There are %d batches in each epoch... " % (batch_count_epoch))
    train(model, dataLoader) 
            
if __name__ == "__main__":
    main()
