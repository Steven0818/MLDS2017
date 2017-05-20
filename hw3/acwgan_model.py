import tensorflow as tf
import numpy as np
import ops
from functools import partial
from gan_model import GAN_model

class ACWGAN_model(GAN_model):
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(64, 64, 3), optimizer_name='RMSProp',
                       eyes_dim = 11, hair_dim = 11, clip_value=(-0.01, 0.01), iter_ratio=5):
        self.clip_value = clip_value
        self.iter_ratio = iter_ratio
        self.eyes_dim = eyes_dim
        self.hair_dim = hair_dim
        GAN_model.__init__(self, z_dim, batch_size, learning_rate, img_shape)
    
    def _create_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool) 
        self.real_imgs = tf.placeholder(tf.float32, [self.batch_size] + list(self.img_shape), name='r_imgs')
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.hair_vec = tf.placeholder(tf.float32, [self.batch_size, self.eyes_dim], name='hair')
        self.eyes_vec = tf.placeholder(tf.float32, [self.batch_size, self.hair_dim], name='eyes')

    def create_computing_graph(self):
        print("Setting up model...")
        
        self._create_placeholder()
        self.gen_images = self._generator(self.z_vec, self.eyes_vec, self.hair_vec, self.train_phase, scope_name='generator')
        
        _, logits_real, _ = self._discriminator(self.real_imgs,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=False)

        _, logits_fake, _ = self._discriminator(self.gen_images,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=True)
        
        logit_eyes_real = self._eyes_classifier(self.real_imgs, 
                                                self.train_phase, 
                                                scope_name='eyes_classifier', 
                                                reuse=False)
        logit_eyes_gene = self._eyes_classifier(self.gen_images, 
                                                self.train_phase, 
                                                scope_name='eyes_classifier', 
                                                reuse=True)
                                                
        logit_hair_real = self._hair_classifier(self.real_imgs, 
                                                self.train_phase, 
                                                scope_name='hair_classifier', 
                                                reuse=False)
        logit_hair_gene = self._hair_classifier(self.gen_images, 
                                                self.train_phase, 
                                                scope_name='hair_classifier', 
                                                reuse=True)

        self._discriminator_loss(logits_real, logits_fake)
        self._generator_loss(logits_fake, None, None)
        e_r_loss = self._classifier_loss(logit_eyes_real, self.eyes_vec)
        e_g_loss = self._classifier_loss(logit_eyes_gene, self.eyes_vec)
        h_r_loss = self._classifier_loss(logit_hair_real, self.hair_vec)
        h_g_loss = self._classifier_loss(logit_hair_gene, self.hair_vec)

        self.eyes_loss = (e_r_loss + e_g_loss) / 2.
        self.eyes_gen_loss = e_g_loss
        self.hair_loss = (h_r_loss + h_g_loss) / 2.
        self.hair_gen_loss = h_g_loss

        self.generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.eyes_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eyes_classifier')
        self.hair_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hair_classifier')
        
        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_e = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_h = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_gen_e = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_gen_h = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        #if self.optimizer_name == 'Adam':
        #    optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        #else:
        #    optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9)

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

        self.eyes_train_op = tf.contrib.layers.optimize_loss(loss=self.eyes_loss, learning_rate=self.learning_rate,
                                                              optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) 
                                                                                if self.optimizer_name == 'Adam' 
                                                                                else tf.train.RMSPropOptimizer, 
                                                              variables=self.eyes_variables, 
                                                              global_step=counter_e)

        self.hair_train_op = tf.contrib.layers.optimize_loss(loss=self.hair_loss, learning_rate=self.learning_rate,
                                                              optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) 
                                                                                if self.optimizer_name == 'Adam' 
                                                                                else tf.train.RMSPropOptimizer, 
                                                              variables=self.hair_variables, 
                                                              global_step=counter_h)
        
        self.eyes_gen_train_op = tf.contrib.layers.optimize_loss(loss=self.eyes_gen_loss, learning_rate=self.learning_rate,
                                                              optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) 
                                                                                if self.optimizer_name == 'Adam' 
                                                                                else tf.train.RMSPropOptimizer, 
                                                              variables=self.generator_variables, 
                                                              global_step=counter_gen_e)

        self.hair_gen_train_op = tf.contrib.layers.optimize_loss(loss=self.hair_gen_loss, learning_rate=self.learning_rate,
                                                              optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) 
                                                                                if self.optimizer_name == 'Adam' 
                                                                                else tf.train.RMSPropOptimizer, 
                                                              variables=self.generator_variables, 
                                                              global_step=counter_gen_h)                                                     




        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.clip_value[0], self.clip_value[1])) for var in self.discriminator_variables]
        # merge the clip operations on critic variables
        with tf.control_dependencies([self.discriminator_train_op]):
            self.discriminator_train_op = tf.tuple(clipped_var_c)

    def _generator(self, z, eyes, hair, train_phase, scope_name='generator'):
        shrink_size = self.pool_size
        with tf.variable_scope(scope_name) as scope:

            init_vec = tf.concat([z, eyes, hair], 1)
            
            train = tf.contrib.layers.fully_connected(init_vec, shrink_size * shrink_size * 1024, 
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

    def _eyes_classifier(self, input_img, train_phase, scope_name='eyes_classifier', reuse=False):
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            img = tf.contrib.layers.conv2d(input_img, num_outputs=64, kernel_size=5, stride=1, 
                                           activation_fn=tf.nn.relu)
            img = tf.contrib.layers.max_pool2d(img, 2, stride=2, padding='SAME')
            img = tf.contrib.layers.conv2d(img, num_outputs=128, kernel_size=5, stride=1, 
                                           activation_fn=tf.nn.relu, 
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           normalizer_params={'is_training':self.train_phase})
            img = tf.contrib.layers.max_pool2d(img, 2, stride=2, padding='SAME')
            img = tf.reshape(img, [self.batch_size, -1])

            eyes_logits = tf.contrib.layers.fully_connected(img, self.eyes_dim, activation_fn=None)
        return eyes_logits

    def _hair_classifier(self, input_img, train_phase, scope_name='eyes_classifier', reuse=False):
        
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            img = tf.contrib.layers.conv2d(input_img, num_outputs=64, kernel_size=5, stride=1, 
                                           activation_fn=tf.nn.relu)
            img = tf.contrib.layers.max_pool2d(img, 2, stride=2, padding='SAME')
            img = tf.contrib.layers.conv2d(img, num_outputs=128, kernel_size=5, stride=1, 
                                           activation_fn=tf.nn.relu, 
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           normalizer_params={'is_training':self.train_phase})
            img = tf.contrib.layers.max_pool2d(img, 2, stride=2, padding='SAME')
            img = tf.reshape(img, [self.batch_size, -1])

            hair_logits = tf.contrib.layers.fully_connected(img, self.hair_dim, activation_fn=None)
        return hair_logits
         
    def _discriminator_loss(self, logits_real, logits_fake):
        self.discriminator_loss = tf.reduce_mean(logits_fake - logits_real)
    
    def _generator_loss(self, logits_fake, feature_fake, feature_real):
        self.generator_loss = tf.reduce_mean(-logits_fake)
        
    def _classifier_loss(self, logits, label):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

    def train_model(self, dataLoader, max_iteration):
        self.global_steps = 0
        self.epoch = 0

        def next_feed_dict(loader, batch_size):
            while True:
                loader.shuffle()
                batch_gen = loader.batch_generator(batch_size=batch_size)
                self.epoch += 1
                for batch_imgs, correct_tag, wrong_tag in batch_gen:
                    batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    feed_dict = {self.z_vec: batch_z,
                                 self.eyes_vec: correct_tag[:,:11], 
                                 self.hair_vec: correct_tag[:,11:], 
                                 self.real_imgs: batch_imgs, 
                                 self.train_phase: True}
                    yield feed_dict

        gen = next_feed_dict(dataLoader, self.batch_size)

        for i in range(max_iteration):
            if self.global_steps < 25 or self.global_steps % 500 == 0:
                iteration = 100
            else:
                iteration = self.iter_ratio
            for it_r in range(iteration):
                self.sess.run(self.discriminator_train_op, feed_dict=next(gen))

            self.sess.run(self.generator_train_op, feed_dict=next(gen))

            self.sess.run(self.eyes_gen_train_op, feed_dict=next(gen))
            self.sess.run(self.hair_gen_train_op, feed_dict=next(gen))
            self.sess.run(self.eyes_train_op, feed_dict=next(gen))
            self.sess.run(self.hair_train_op, feed_dict=next(gen))
            
            if self.global_steps % 20 == 0 and self.global_steps != 0:
                g_loss_val, d_loss_val, e_loss_val, h_loss_val = self.sess.run(
                    [self.generator_loss, self.discriminator_loss, self.eyes_loss, self.hair_loss], feed_dict=next(gen))
                print("Epoch %d, Step: %d, generator loss: %g, discriminator_loss: %g, eyes_loss: %g, hair_loss: %g" 
                                % (i, self.global_steps, g_loss_val, d_loss_val, e_loss_val, h_loss_val))
                            
                self.global_steps += 1  
            
            if i % 1000 == 0 and i != 0:
                self.saver.save(self.sess, "acwmodel/acwmodel_%d.ckpt" % self.global_steps, global_step=self.global_steps)
                self._visualize_model('acwresult')            
        
    def _visualize_model(self, img_dir):
        print("Sampling images from model...")
        
        batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        correct_tag = np.zeros([self.batch_size, self.eyes_dim + self.hair_dim], dtype=np.float32)
        correct_tag[:,1] = 1.
        correct_tag[:-1] = 1.
        feed_dict = {self.z_vec: batch_z, self.eyes_vec: correct_tag[:,:11], self.hair_vec: correct_tag[:,11:], self.train_phase: False}

        images = self.sess.run(self.gen_images, feed_dict=feed_dict)
        images = ops.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        shape = [4, self.batch_size // 4]
        
        print(images.shape)
        ops.save_imshow_grid(images, img_dir, "generated_%d.png" % self.global_steps, shape)


