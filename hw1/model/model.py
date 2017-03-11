import tensorflow as tf
import numpy as np
import GRU_module as GRU


from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

class model(self):
	def __init__(self,hyperPara_info):
		###GLOBAL STEP###
		global_step = tf.Variable(0, trainable=False)

		###HYPER PARAMETER###
		batchSize = hyperPara_info['batchSize']
		learningRate = hyperPara_info['learningRate']
		SentenceLen = hyperPara_info['sentenceLen']
		xDim = hyperPara_info['xDim']
		numClass = hyperPara_info['numClass'] #number of vocabulary
		embeddedSize = hyperPara_info['gru']['output_size']

		###VARIABLE###
		tf.variable_scope('NCE'):
			nce_weights = tf.get_variable("weights", [numClass,embeddedSize],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
			nce_bias = tf.get_variable("bias", [numClass],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

		###PLACEHOLDER###
		self.x = tf.placeholder(shape=(batchSize,SentenceLen,xDim),dtype=tf.float32)
		self.x_index = tf.placeholder(shape=(batchSize,SentenceLen),dtype=tf.float32)
		
		###MODEL###
		gru = GRU.BiGRU(	
						inputs=self.x,
						num_layers=hyperPara_info['gru']['num_layers'],
						num_hidden_neurons=hyperPara_info['gru']['num_hidden_neurons'],
						output_size=embeddedSize,
						dropout=hyperPara_info['gru']['dropout'],
						activation=hyperPara_info['gru']['activation'],
						name=hyperPara_info['gru']['name'],
						sequence_length=None,
						reuse=False
					) #(batchSize,SentenceLen,embeddedSize)

		gru_output = tf.reshape(gru['output'][:,2:SentenceLen-1,:],[batchSize*(SentenceLen-3),embeddedSize])

		label = tf.reshape(self.x_index[:,3:],[batchSize*(SentenceLen-3),1])

		
		self.loss = tf.reduce_mean(
						tf.nn.sampled_softmax_loss(
							weights=nce_weights,	#[num_classes, dim]
                   			biases=nce_bias,		#[num_classes]
                   			labels=label,			#[batch_size, num_true]
                   			inputs=gru_output,		#[batch_size, dim]
                   			num_sampled=hyperPara_info['num_sampled'],#int
                   			num_classes=numClass   	#int 30000
                   			)
						)

		self.result = tf.nn.softmax(tf.matmul(gru_output, tf.transpose(nce_weights)) + nce_bias) ##(batchSize*(SentenceLen-3), numClass)
		self.predict = tf.reshape(tf.argmax(self.result,1),[batchSize,SentenceLen-3])
		
        self.train_op = AdamaxOptimizer(learning_rate = learningRate).minimize(self.loss,global_step=global_step)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))


    def initialize(self,):
    	self.sess.run(tf.global_variables_initializer()) 


	def train(self,x,x_index):
		_,loss = self.sess.run([self.train_op, self.loss], feed_dict={self.x:x,self.x_index:x_index})


	def predict(self,):
		result = self.sess.run(self.predict, feed_dict={self.x:x,self.x_index:x_index})