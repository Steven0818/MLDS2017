import json 
import numpy as np
import model
import utility

datas = json.load(open('../data/data_40.json','r'))
print len(datas)
test_datas = json.load(open('../data/test.json','r'))


BATCH_SIZE = 20
EPOCH = 100
HYPER_INFO = {
	'batchSize':20,
	'learningRate':0.001,
	'sentenceLen':40,
	'xDim':30000,
	'numClass':30000,
	'gru':{
		'output_size':3000,
		'num_layers':2,
		'num_hidden_neurons':300,
		'dropout':0.5,
		'activation':'relu',
		'name':'GRU'
	},
	'num_sampled':3000
}

print ("model building....")
train_model = model.model(HYPER_INFO)

print ("parameter initialize....")
train_model.initialize()

print ("start training...")
utility.util_train(train_model,datas,test_datas,HYPER_INFO['batchSize'],EPOCH,HYPER_INFO['sentenceLen'],HYPER_INFO['xDim'])