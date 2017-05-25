import tensorflow as tf
import numpy as np
import ops
from functools import partial
from gan_model import GAN_model

class conditional_WGAN_model(GAN_model):
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(64, 64, 3), optimizer_name='RMSProp',
                       tag_dim = 22, tag_embed_dim = 100, clip_value=(-0.01, 0.01), iter_ratio=5):
        self.clip_value = clip_value
        self.iter_ratio = iter_ratio
        self.tag_dim = tag_dim
        self.tag_embed_dim = tag_embed_dim
        GAN_model.__init__(self, z_dim, batch_size, learning_rate, img_shape)
    
    def _create_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool) 
        self.real_imgs = tf.placeholder(tf.float32, [self.batch_size] + list(self.img_shape), name='r_imgs')
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.tag_vec = tf.placeholder(tf.float32, [self.batch_size, self.tag_dim], name='tag')
        self.wtag_vec = tf.placeholder(tf.float32, [self.batch_size, self.tag_dim], name='w_tag')

    def create_computing_graph(self):
        print("Setting up model...")
        
        self._create_placeholder()
        self.gen_images = self._generator(self.z_vec, self.tag_vec, self.train_phase, scope_name='generator')
        
        _, logits_real, _ = self._discriminator(self.real_imgs,
                                                self.tag_vec,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=False)
        
        _, logits_wtag, _ = self._discriminator(self.real_imgs,
                                                self.wtag_vec,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=True)

        _, logits_fake, _ = self._discriminator(self.gen_images,
                                                self.tag_vec,
                                                self.train_phase,
                                                scope_name="discriminator",
                                                reuse=True)

        self._discriminator_loss(logits_real, logits_fake, logits_wtag)
        self._generator_loss(logits_fake, logits_wtag, None, None)
        
        
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

    def _generator(self, z, tag_vec, train_phase, scope_name='generator'):
        shrink_size = self.pool_size
        with tf.variable_scope(scope_name) as scope:
            
            init_vec = tf.concat([z, tag_vec], 1)
            
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
            
    def _discriminator(self, input_img, tag_vec, train_phase, scope_name='discriminator', reuse=False):
        
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
            
            tag_embedding = tf.contrib.layers.fully_connected(tag_vec, self.tag_embed_dim, activation_fn=ops.leaky_relu)
            tag_embedding = tf.expand_dims(tag_embedding, 1)
            tag_embedding = tf.expand_dims(tag_embedding, 2)
            tiled_embedding = tf.tile(tag_embedding, [1, 4, 4, 1])
            
            chn_concat = tf.concat([img, tiled_embedding], 3)
            new_img = tf.contrib.layers.conv2d(chn_concat, num_outputs=512, kernel_size=1,
                                               activation_fn=ops.leaky_relu, 
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               normalizer_params={'is_training':self.train_phase})

            logit = tf.contrib.layers.fully_connected(tf.reshape(new_img, [self.batch_size, -1]), 1, activation_fn=None)
                
        return None, logit, None
         
         
    def _discriminator_loss(self, logits_real, logits_fake, logits_wtag):
        self.discriminator_loss = tf.reduce_mean((logits_wtag + logits_fake) / 2 - logits_real)
        
    
    def _generator_loss(self, logits_fake, logits_wtag, feature_fake, feature_real):
        self.generator_loss = tf.reduce_mean((-logits_fake))
        
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
                                 self.tag_vec: correct_tag, 
                                 self.wtag_vec: wrong_tag, 
                                 self.real_imgs: batch_imgs, 
                                 self.train_phase: True}
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
        
            if i % 1000 == 0 and i != 0:
                self.saver.save(self.sess, "cwmodel/cwmodel_%d.ckpt" % self.global_steps, global_step=self.global_steps)
                self._visualize_model('cwresult')            
        
    def _visualize_model(self, img_dir):
        print("Sampling images from model...")
        
        batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        correct_tag = np.zeros([self.batch_size, self.tag_dim], dtype=np.float32)
        correct_tag[:,1] = 1.
        correct_tag[:-1] = 1.
        feed_dict = {self.z_vec: batch_z, self.tag_vec: correct_tag, self.train_phase: False}

        images = self.sess.run(self.gen_images, feed_dict=feed_dict)
        images = ops.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        shape = [4, self.batch_size // 4]
        
        print(images.shape)
        ops.save_imshow_grid(images, img_dir, "generated_%d.png" % self.global_steps, shape)


class conditional_GAN_model(GAN_model):
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(64, 64, 3), optimizer_name='Adam',
                       tag_dim = 4800, tag_embed_dim = 100, test_feature = 'test2.npy'):
        self.tag_dim = tag_dim
        self.tag_embed_dim = tag_embed_dim
        self.test_feature = np.repeat(np.load(test_feature), batch_size, axis=0)
        GAN_model.__init__(self, z_dim, batch_size, learning_rate, img_shape)
    
    def _create_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool) 
        self.real_imgs = tf.placeholder(tf.float32, [self.batch_size] + list(self.img_shape), name='r_imgs')
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.tag_vec = tf.placeholder(tf.float32, [self.batch_size, self.tag_dim], name='tag')
        self.wtag_vec = tf.placeholder(tf.float32, [self.batch_size, self.tag_dim], name='w_tag')

    def create_computing_graph(self):
        print("Setting up model...")
        
        self._create_placeholder()
        self.gen_images = self._generator(self.z_vec, self.tag_vec, self.train_phase, scope_name='generator')
        process_image = self.gen_images*127.5 + 127.5
        self.summary_image = tf.summary.image('gen_images', process_image, max_outputs=20)
        self.summary_image_test = tf.summary.image('test_gen_images', process_image, max_outputs=20)
        logits_real = self._discriminator(self.real_imgs,
                                          self.tag_vec,
                                          self.train_phase,
                                          scope_name="discriminator",
                                          reuse=False)

        logits_wtag = self._discriminator(self.real_imgs,
                                          self.wtag_vec,
                                          self.train_phase,
                                          scope_name="discriminator",
                                          reuse=True)

        logits_fake = self._discriminator(self.gen_images,
                                          self.tag_vec,
                                          self.train_phase,
                                          scope_name="discriminator",
                                          reuse=True)

        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        self._discriminator_loss(logits_real, logits_fake, logits_wtag)
        self._generator_loss(logits_fake)
     
        self.generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        

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
                                            
    def _generator(self, z, tag_vec, train_phase, scope_name='generator'):
        shrink_size = self.pool_size
        with tf.variable_scope(scope_name) as scope:
            reduced_text_embedding = ops.lrelu(ops.linear(tag_vec, self.tag_embed_dim, 'g_embedding'))

            init_vec = tf.concat([z, reduced_text_embedding], 1)

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
            
    def _discriminator(self, input_img, tag_vec, train_phase, scope_name='discriminator', reuse=False):
        
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
            
            tag_embedding = ops.lrelu(ops.linear(tag_vec, self.tag_embed_dim, 'd_embedding'))
            tag_embedding = tf.expand_dims(tag_embedding, 1)
            tag_embedding = tf.expand_dims(tag_embedding, 2)
            tiled_embedding = tf.tile(tag_embedding, [1, 4, 4, 1])
            
            chn_concat = tf.concat([img, tiled_embedding], 3)
            new_img = tf.contrib.layers.conv2d(chn_concat, num_outputs=512, kernel_size=1,
                                               activation_fn=ops.leaky_relu, 
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               normalizer_params={'is_training':self.train_phase})

            logit = tf.contrib.layers.fully_connected(tf.reshape(new_img, [self.batch_size, -1]), 1, activation_fn=None)
                
        return logit
         
         
    def _discriminator_loss(self, logits_real, logits_fake, logits_wtag):
        real_labels = tf.ones_like(logits_real)
        real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=logits_real))
        
        fake_labels = tf.zeros_like(logits_fake)
        fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=logits_fake))

        wtag_labels = tf.zeros_like(logits_wtag)
        wtag_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=wtag_labels, logits=logits_wtag))

        self.discriminator_loss = real_d_loss + fake_d_loss + wtag_d_loss
        
        d_loss = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
        r_d_loss = tf.summary.scalar('read_d_loss', real_d_loss)
        f_d_loss = tf.summary.scalar('fake_d_loss', fake_d_loss)
        w_d_loss = tf.summary.scalar('wtag_d_loss', wtag_d_loss)

        self.summary_d = tf.summary.merge([d_loss, r_d_loss, f_d_loss, w_d_loss])
    
    def _generator_loss(self, logits_fake):
        fake_labels = tf.ones_like(logits_fake)
        
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=logits_fake))
        self.summary_g = tf.summary.scalar('generater_loss', self.generator_loss)
        
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
                                 self.tag_vec: correct_tag, 
                                 self.wtag_vec: wrong_tag, 
                                 self.real_imgs: batch_imgs, 
                                 self.train_phase: True}
                    yield feed_dict

        gen = next_feed_dict(dataLoader, self.batch_size)
        
        for i in range(max_iteration):

            _, d_loss_val, summary_d = self.sess.run([self.discriminator_train_op, self.discriminator_loss, self.summary_d], feed_dict=next(gen))

            self.summary_writer.add_summary(summary_d, self.global_steps)

            _, g_loss_val = self.sess.run([self.generator_train_op, self.generator_loss], feed_dict=next(gen)) 
            _, g_loss_val, summary_g, summary_image = self.sess.run([self.generator_train_op, self.generator_loss, self.summary_g, self.summary_image],feed_dict=next(gen))

            self.summary_writer.add_summary(summary_g, self.global_steps)
            
            if self.global_steps % 20 == 0 and self.global_steps != 0:
                self.summary_writer.add_summary(summary_image, self.global_steps)
                self.summary_writer.flush()
                print("Epoch %d, Step: %d, generator loss: %g, discriminator_loss: %g" % (i, self.global_steps, g_loss_val, d_loss_val))

            self.global_steps += 1
        
            if i % 200 == 0 and i != 0:
                self._visualize_model('cresult')            
                self.saver.save(self.sess, "cmodel/cmodel_%d.ckpt" % self.global_steps, global_step=self.global_steps)
        
    def _visualize_model(self, img_dir):
        print("Sampling images from model...")
        
        batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)

        feed_dict = {self.z_vec: batch_z, self.tag_vec: self.test_feature, self.train_phase: False}

        images, summary_img = self.sess.run([self.gen_images, self.summary_image_test], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_img, self.global_steps)
        images = ops.unprocess_image(images, 127.5, 127.5).astype(np.uint8)
        shape = [4, self.batch_size // 4]

        print(images.shape)
        ops.save_imshow_grid(images, img_dir, "generated_%d.png" % self.global_steps, shape)
    
    def test_single_feature(self, feature):
        batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        batch_feature = feature

        feed_dict = {self.z_vec: batch_z, self.tag_vec: batch_feature, self.train_phase: False}
        images, summary_image = self.sess.run([self.gen_images, self.summary_image], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_image)
        self.summary_writer.flush()

    def test_loader(self, data_loader):
        def next_feed_dict(loader, batch_size):
            while True:
                loader.shuffle()
                batch_gen = loader.batch_generator(batch_size=batch_size)
                for batch_imgs, correct_tag, wrong_tag in batch_gen:
                    batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    feed_dict = {self.z_vec: batch_z, 
                                 self.tag_vec: correct_tag, 
                                 self.train_phase: False}
                    yield feed_dict
        gen = next_feed_dict(data_loader, self.batch_size)
        images, summary_image = self.sess.run(
            [self.gen_images, self.summary_image], feed_dict=next(gen))
        self.summary_writer.add_summary(summary_image)
        self.summary_writer.flush()

    def load_model(self, model_path):
        saver = tf.train.Saver(restore_sequentially=True)
        saver.restore(self.sess, model_path)

