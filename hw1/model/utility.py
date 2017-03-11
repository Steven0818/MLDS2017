import numpy as np
from random import shuffle
import csv


def index2vector(input_data,max_sentence_len,word_dimension):
	
	sentences = np.zeros([len(input_data),max_sentence_len,word_dimension])
	
	for i,data in enumerate(input_data):
		for j,one_hot_index in enumerate(data):
			sentences[i,j,one_hot_index] = 1

	return sentences


def test(model,test_datas,batch_size,max_sentence_len,word_dimension,global_step):
	
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
			predict_result = model.predict(input_sentences,input_data)  
			
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
		predict_result = model.predict(input_sentences,input_data)  
			
		for i,predict_distribute in enumerate(predict_result):
			answer_prob_distribute = np.asarray(predict_distribute[answer_word_index[i]-1])[answer_option[i]]
			question_count+=1
			predict_pair = [question_count,answer_list[np.argmax(answer_prob_distribute)]]
			predict_list.append(predict_pair)


	f = open("csv/test_"+str(global_step)+".csv","w")  
	w = csv.writer(f)  
	w.writerows(predict_list)  
	print ("finish prediction.....")


def util_train(model,datas,test_datas,batch_size,epoch,max_sentence_len,word_dimension):
	
	global_step = 0
	for i in range(epoch):
		totalcost_epoch = 0
		batchcount = 0  
		input_data = []
		shuffle(datas)
		for data in datas:
			input_data.append(data)
			if len(input_data)%batch_size == 0:
				batchcount+=1
				global_step+=1
				input_sentences = index2vector(input_data,max_sentence_len,word_dimension)
				cost = model.train(input_sentences,input_data)
				print ('global_step '+str(global_step)+' ,Cost of Epoch ' + str(i) + ' batch ' + str(batchcount) + ": "+str(cost))
				totalcost_epoch+=cost
				if global_step%200 == 0:
					test(model,test_datas,batch_size,max_sentence_len,word_dimension,global_step)
				input_data = []
		print ('Cost of Epoch ' + str(i) + ": "+str(totalcost_epoch/batchcount))
