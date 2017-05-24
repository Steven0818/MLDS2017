import tensorflow as tf
import numpy as np
import ops
from functools import partial

class GAN_model():
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(64, 64, 3), optimizer_name='Adam'):
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_shape = img_shape
        self.pool_size = img_shape[0] // (2**4)
        self.optimizer_name = optimizer_name
    
    def create_computing_graph(self):
        print("Setting up model...")
        
        self._create_placeholder()
        self.gen_images = self._generator(self.z_vec, self.train_phase, scope_name='generator')
        
        # re-initialize discriminator parameter when training discriminator
        discr_real_prob, logits_real, feature_real = self._discriminator(self.real_imgs,
                                                                         self.train_phase,
                                                                         scope_name="discriminator",
                                                                         reuse=False)
        
        # reuse discriminator parameter when training generator
        discr_fake_prob, logits_fake, feature_fake = self._discriminator(self.gen_images,
                                                                         self.train_phase,
                                                                         scope_name="discriminator",
                                                                         reuse=True)

        self._discriminator_loss(logits_real, logits_fake)
        self._generator_loss(logits_fake, feature_fake, feature_real)
        
        train_variables = tf.trainable_variables()
        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        
        if self.optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        else:
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            generator_grads = optimizer.compute_gradients(self.generator_loss, var_list=self.generator_variables)
            discriminator_grad = optimizer.compute_gradients(self.discriminator_loss, self.discriminator_variables)
            self.generator_train_op = optimizer.apply_gradients(generator_grads)
            self.discriminator_train_op = optimizer.apply_gradients(discriminator_grad)
    
    def _create_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool)
        self.real_imgs = tf.placeholder(tf.float32, [self.batch_size] + list(self.img_shape), name='r_imgs')
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        
    def _generator(self, z, train_phase, scope_name='generator'):
        shrink_size = self.pool_size
        with tf.variable_scope(scope_name) as scope:
            # z_fc_layer(name, input, n_neuron, img_size, add_bias, activation, train_phase)
            # output_shape = (4, 4, 1024)
            h = ops.z_fc_layer('z', self.z_vec, 
                               shrink_size*shrink_size*1024, 
                               shrink_size, 
                               add_bias=True, 
                               activation=tf.nn.relu, 
                               train_phase=self.train_phase)
            
            shrink_size *= 2
            # transpose_conv_layer(name, input , ksize, output_shape, strides, add_bias, activation, train_phase), 
            # output_shape = (8, 8, 512)
            h = ops.transpose_conv_layer('conv1', h,
                                         [5 ,5 , 512, 1024], 
                                         [self.batch_size, shrink_size, shrink_size, 512], 
                                         strides=[1,2,2,1], 
                                         add_bias=True,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase )
            
            #output_shape = (16, 16, 256)
            shrink_size *= 2
            h = ops.transpose_conv_layer('conv2', h, 
                                         [5 ,5 , 256, 512], 
                                         [self.batch_size, shrink_size, shrink_size, 256], 
                                         strides=[1,2,2,1], 
                                         add_bias=True,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase)
            
            #output_shape = (32, 32, 128)
            shrink_size *= 2
            h = ops.transpose_conv_layer('conv3', h, 
                                         [5 ,5 , 128, 256], 
                                         [self.batch_size, shrink_size, shrink_size, 128],
                                         strides=[1,2,2,1],
                                         add_bias=True,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase)
            
            #output_shape = (64, 64, 3)
            shrink_size *= 2
            gen_img = ops.transpose_conv_layer('gen', h, 
                                               [5 ,5 , 3, 128],
                                               [self.batch_size, shrink_size, shrink_size, 3],
                                               strides=[1,2,2,1], 
                                               add_bias=True,
                                               activation=tf.nn.tanh, 
                                               train_phase=self.train_phase)
            
        return gen_img
            
    def _discriminator(self, input_img, train_phase, scope_name='discriminator', reuse=False):
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            
            #       conv_layer(name, input , ksize, strides, do_bn, activation, train_phase)
            h = ops.conv_layer('conv1', input_img, 
                               [5, 5, 3,64], 
                               strides=[1,2,2,1], 
                               add_bias=True,
                               do_bn=False, 
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)
                               
            h = ops.conv_layer('conv2', h, 
                               [5, 5, 64, 128], 
                               strides=[1,2,2,1], 
                               add_bias=True,
                               do_bn=True, 
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)

            h = ops.conv_layer('conv3', h, 
                               [5, 5, 128, 256], 
                               strides=[1,2,2,1], 
                               add_bias=True, 
                               do_bn=True,
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)

            h = ops.conv_layer('conv4', h, 
                               [5, 5, 256, 512], 
                               [1,2,2,1], 
                               add_bias=True,
                               do_bn=True,
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)
            
            flatten = tf.reshape(h, [self.batch_size, -1])
            
            #            rf_fc_layer(name, input, n_neuron)
            h_pred = ops.rf_fc_layer('discr', flatten, 1, add_bias=True)

        return tf.nn.sigmoid(h_pred), h_pred, flatten
         
         
    def _discriminator_loss(self, logits_real, logits_fake):
        real_labels = tf.ones_like(logits_real)
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=logits_real))
        
        fake_labels = tf.zeros_like(logits_fake)
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=logits_fake))
        
        self.discriminator_loss = real_loss + fake_loss
        
    
    def _generator_loss(self, logits_fake, feature_fake, feature_real):
        gen_labels = tf.ones_like(logits_fake)
        gen_discr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gen_labels, logits=logits_fake))
        
        #gen_feat_loss = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.pool_size*self.pool_size*512)
        
        #self.generator_loss = gen_discr_loss + 0.1 * gen_feat_loss
        self.generator_loss = gen_discr_loss
    
    
    def _train_op(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)
        
    def initialize_network(self):
        print("Initializing network...")
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter('./summary')
        
    def train_model(self, dataLoader, max_epoch):
        self.global_steps = 0
        
        for i in range(max_epoch):
            dataLoader.shuffle() 
            batch_gen = dataLoader.batch_generator(batch_size=self.batch_size)

            for batch_imgs in batch_gen:
                batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                feed_dict = {self.z_vec: batch_z, self.real_imgs: batch_imgs, self.train_phase: True}
    
                self.sess.run(self.discriminator_train_op, feed_dict=feed_dict)
                self.sess.run(self.generator_train_op, feed_dict=feed_dict)
    
                if self.global_steps % 20 == 0 and self.global_steps != 0:
                    g_loss_val, d_loss_val = self.sess.run(
                        [self.generator_loss, self.discriminator_loss], feed_dict=feed_dict)
                    print("Epoch %d, Step: %d, generator loss: %g, discriminator_loss: %g" % (i, self.global_steps, g_loss_val, d_loss_val))
                    
                self.global_steps += 1
            
            if i % 4 == 0 and i != 0:
                self.saver.save(self.sess, "model/model_%d.ckpt" % self.global_steps, global_step=self.global_steps)
                self._visualize_model('result')    
            
    def _visualize_model(self, img_dir):
        print("Sampling images from model...")
        
        batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        feed_dict = {self.z_vec: batch_z, self.train_phase: False}

        images = self.sess.run(self.gen_images, feed_dict=feed_dict)
        images = ops.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        shape = [4, self.batch_size // 4]
        
        print(images.shape)
        ops.save_imshow_grid(images, img_dir, "generated_%d.png" % self.global_steps, shape)

class WGAN_model2(GAN_model):
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(64, 64, 3), optimizer_name='RMSProp',
                       clip_value=(-0.01, 0.01), iter_ratio=5):
        self.clip_value = clip_value
        self.iter_ratio = iter_ratio
        GAN_model.__init__(self, z_dim, batch_size, learning_rate, img_shape)
    
    def create_computing_graph(self):
        print("Setting up model...")
        
        self._create_placeholder()
        self.gen_images = self._generator(self.z_vec, self.train_phase, scope_name='generator')
        
        # re-initialize discriminator parameter when training discriminator
        _, logits_real, _ = self._discriminator(self.real_imgs,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=False)
        
        # reuse discriminator parameter when training generator
        _, logits_fake, _ = self._discriminator(self.gen_images,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=True)

        self._discriminator_loss(logits_real, logits_fake)
        self._generator_loss(logits_fake, None, None)
        
        
        self.generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        
        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        if self.optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        else:
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9)

        self.generator_train_op = tf.contrib.layers.optimize_loss(loss=self.generator_loss, learning_rate=self.learning_rate,
                                                                  optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) 
                                                                                    if self.optimizer_name == 'Adam' 
                                                                                    else tf.train.RMSPropOptimizer, 
                                                                  variables=self.generator_variables, 
                                                                  global_step=counter_g)
    
        self.discriminator_train_op = tf.contrib.layers.optimize_loss(loss=self.discriminator_loss, learning_rate=self.learning_rate,
                                                                      optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) 
                                                                                        if self.optimizer_name == 'Adam' 
                                                                                        else tf.train.RMSPropOptimizer, 
                                                                      variables=self.discriminator_variables, 
                                                                      global_step=counter_d)
                                            
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.clip_value[0], self.clip_value[1])) for var in self.discriminator_variables]
        # merge the clip operations on critic variables
        with tf.control_dependencies([self.discriminator_train_op]):
            self.discriminator_train_op = tf.tuple(clipped_var_c)

    def _generator(self, z, train_phase, scope_name='generator'):
        shrink_size = self.pool_size
        with tf.variable_scope(scope_name) as scope:
            train = tf.contrib.layers.fully_connected(z, shrink_size * shrink_size * 1024, 
                                                      activation_fn=ops.leaky_relu, 
                                                      normalizer_fn=tf.contrib.layers.batch_norm)
            train = tf.reshape(train, (-1, shrink_size, shrink_size, 1024))
            train = tf.contrib.layers.conv2d_transpose(train, 512, 5, stride=2,
                                        activation_fn=tf.nn.relu, 
                                        normalizer_fn=tf.contrib.layers.batch_norm, padding='SAME', 
                                        weights_initializer=tf.random_normal_initializer(0, 0.02),
                                        normalizer_params={'is_training':self.train_phase})
            train = tf.contrib.layers.conv2d_transpose(train, 256, 5, stride=2,
                                        activation_fn=tf.nn.relu, 
                                        normalizer_fn=tf.contrib.layers.batch_norm, padding='SAME', 
                                        weights_initializer=tf.random_normal_initializer(0, 0.02), 
                                        normalizer_params={'is_training':self.train_phase})
            train = tf.contrib.layers.conv2d_transpose(train, 128, 5, stride=2,
                                        activation_fn=tf.nn.relu, 
                                        normalizer_fn=tf.contrib.layers.batch_norm, padding='SAME', 
                                        weights_initializer=tf.random_normal_initializer(0, 0.02),
                                        normalizer_params={'is_training':self.train_phase})
            gen_img = tf.contrib.layers.conv2d_transpose(train, 3, 5, stride=2, 
                                        activation_fn=tf.nn.tanh, 
                                        normalizer_fn=tf.contrib.layers.batch_norm, padding='SAME', 
                                        weights_initializer=tf.random_normal_initializer(0, 0.02),
                                        normalizer_params={'is_training':self.train_phase})
        return gen_img
            
    def _discriminator(self, input_img, train_phase, scope_name='discriminator', reuse=False):
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            
            img = tf.contrib.layers.conv2d(input_img, num_outputs=64, kernel_size=5, stride=2, 
                                           activation_fn=ops.leaky_relu)
            img = tf.contrib.layers.conv2d(img, num_outputs=128, kernel_size=5, stride=2, 
                                           activation_fn=ops.leaky_relu, 
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           normalizer_params={'is_training':self.train_phase})
            img = tf.contrib.layers.conv2d(img, num_outputs=256, kernel_size=5, stride=2,
                                           activation_fn=ops.leaky_relu, 
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           normalizer_params={'is_training':self.train_phase})                            
            img = tf.contrib.layers.conv2d(img, num_outputs=512, kernel_size=5, stride=2,
                                           activation_fn=ops.leaky_relu, 
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           normalizer_params={'is_training':self.train_phase})
            logit = tf.contrib.layers.fully_connected(tf.reshape(img, [self.batch_size, -1]), 1, activation_fn=None)
                
        return None, logit, None
         
         
    def _discriminator_loss(self, logits_real, logits_fake):
        self.discriminator_loss = tf.reduce_mean(logits_fake-logits_real)
        
    
    def _generator_loss(self, logits_fake, feature_fake, feature_real):
        self.generator_loss = tf.reduce_mean(-logits_fake)
        
    def train_model(self, dataLoader, max_iteration):
        self.global_steps = 0
        self.epoch = 0

        def next_feed_dict(loader, batch_size):
            while True:
                loader.shuffle()
                batch_gen = loader.batch_generator(batch_size=batch_size)
                self.epoch += 1
                for batch_imgs in batch_gen:
                    batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    feed_dict = {self.z_vec: batch_z, self.real_imgs: batch_imgs, self.train_phase: True}

                    yield feed_dict
        
        gen = next_feed_dict(dataLoader, self.batch_size)
        
        for i in range(max_iteration):
            
            if self.global_steps < 25 or self.global_steps % 500 == 0:
                iteration = 40
            else:
                iteration = self.iter_ratio

            for it_r in range(iteration):
                self.sess.run(self.discriminator_train_op, feed_dict=next(gen))
                
            self.sess.run(self.generator_train_op, feed_dict=next(gen))
    
            if self.global_steps % 20 == 0 and self.global_steps != 0:
                g_loss_val, d_loss_val = self.sess.run(
                    [self.generator_loss, self.discriminator_loss], feed_dict=next(gen))
                print("Epoch %d, Step: %d, generator loss: %g, discriminator_loss: %g" % (i, self.global_steps, g_loss_val, d_loss_val))
                
            self.global_steps += 1
        
            if i % 600 == 0 and i != 0:
                self.saver.save(self.sess, "model/wmodel_%d.ckpt" % self.global_steps, global_step=self.global_steps)
                self._visualize_model('wresult')            
