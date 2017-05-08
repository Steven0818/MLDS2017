import tensorflow as tf
import numpy as np
import ops

class GAN_model():
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(96, 96, 3), optimizer_name='Adam'):
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
        
        # reuse discriminator parameter when training generetor
        discr_fake_prob, logits_fake, feature_fake = self._discriminator(self.gen_images,
                                                                                 self.train_phase,
                                                                                 scope_name="discriminator",
                                                                                 reuse=True)

        self._discriminator_loss(logits_real, logits_fake)
        self._generetor_loss(logits_fake, feature_fake, feature_real)
        
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
            # output_shape = (1024, 6, 6)
            h = ops.z_fc_layer('z', self.z_vec, 
                               shrink_size*shrink_size*1024, 
                               shrink_size, 
                               add_bias=True, 
                               activation=tf.nn.relu, 
                               train_phase=self.train_phase)
            
            shrink_size *= 2
            # transpose_conv_layer(name, input , ksize, output_shape, strides, add_bias, activation, train_phase), 
            # output_shape = (512, 12, 12)
            h = ops.transpose_conv_layer('conv1', h,
                                         [5 ,5 , 512, 1024], 
                                         [self.batch_size, shrink_size, shrink_size, 512], 
                                         strides=[1,2,2,1], 
                                         add_bias=True,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase )
            
            #output_shape = (256, 24, 24)
            shrink_size *= 2
            h = ops.transpose_conv_layer('conv2', h, 
                                         [5 ,5 , 256, 512], 
                                         [self.batch_size, shrink_size, shrink_size, 256], 
                                         strides=[1,2,2,1], 
                                         add_bias=True,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase)
            
            #output_shape = (128, 48, 48)
            shrink_size *= 2
            h = ops.transpose_conv_layer('conv3', h, 
                                         [5 ,5 , 128, 256], 
                                         [self.batch_size, shrink_size, shrink_size, 128],
                                         strides=[1,2,2,1],
                                         add_bias=True,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase)
            
            #output_shape = (64, 96, 96)
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
        
    
    def _generetor_loss(self, logits_fake, feature_fake, feature_real):
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

class WGAN_model(GAN_model):
    def __init__(self, z_dim=100, batch_size=100, learning_rate=0.0002, img_shape=(96, 96, 3), optimizer_name='RMSProp',
                       clip_value=(-0.01, 0.01), iter_ratio=5):
        self.clip_value = clip_value
        self.iter_ratio = iter_ratio
        GAN_model.__init__(self, z_dim, batch_size, learning_rate, img_shape, optimizer_name)
    
    def _generator(self, z, train_phase, scope_name='generator'):
        shrink_size = self.pool_size
        with tf.variable_scope(scope_name) as scope:
            # z_fc_layer(name, input, n_neuron, img_size, add_bias, activation, train_phase)
            # output_shape = (1024, 6, 6)
            h = ops.z_fc_layer('z', self.z_vec, 
                               shrink_size*shrink_size*1024, 
                               shrink_size, 
                               add_bias=False, 
                               activation=tf.nn.relu, 
                               train_phase=self.train_phase)
            shrink_size *= 2
            # transpose_conv_layer(name, input , ksize, output_shape, strides, add_bias, activation, train_phase), 
            # output_shape = (512, 12, 12)
            h = ops.transpose_conv_layer('conv1', h,
                                         [5 ,5 , 512, 1024], 
                                         [self.batch_size, shrink_size, shrink_size, 512], 
                                         strides=[1,2,2,1], 
                                         add_bias=False,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase )
            
            #output_shape = (256, 24, 24)
            shrink_size *= 2
            h = ops.transpose_conv_layer('conv2', h, 
                                         [5 ,5 , 256, 512], 
                                         [self.batch_size, shrink_size, shrink_size, 256], 
                                         strides=[1,2,2,1], 
                                         add_bias=False,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase)
            
            #output_shape = (128, 48, 48)
            shrink_size *= 2
            h = ops.transpose_conv_layer('conv3', h, 
                                         [5 ,5 , 128, 256], 
                                         [self.batch_size, shrink_size, shrink_size, 128],
                                         strides=[1,2,2,1],
                                         add_bias=False,
                                         activation=tf.nn.relu, 
                                         train_phase=self.train_phase)
            
            #output_shape = (64, 96, 96)
            shrink_size *= 2
            gen_img = ops.transpose_conv_layer('gen', h, 
                                               [5 ,5 , 3, 128],
                                               [self.batch_size, shrink_size, shrink_size, 3],
                                               strides=[1,2,2,1], 
                                               add_bias=False,
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
                               add_bias=False,
                               do_bn=False, 
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)
                               
            h = ops.conv_layer('conv2', h, 
                               [5, 5, 64, 128], 
                               strides=[1,2,2,1], 
                               add_bias=False,
                               do_bn=True, 
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)

            h = ops.conv_layer('conv3', h, 
                               [5, 5, 128, 256], 
                               strides=[1,2,2,1], 
                               add_bias=False, 
                               do_bn=True,
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)

            h = ops.conv_layer('conv4', h, 
                               [5, 5, 256, 512], 
                               [1,2,2,1], 
                               add_bias=False,
                               do_bn=True,
                               activation=ops.leaky_relu, 
                               train_phase=self.train_phase)
            
            flatten = tf.reshape(h, [self.batch_size, -1])
            
            #            rf_fc_layer(name, input, n_neuron)
            h_pred = ops.rf_fc_layer('discr', flatten, 1, add_bias=False)

        return None, h_pred, None
         
         
    def _discriminator_loss(self, logits_real, logits_fake):
        self.discriminator_loss = tf.reduce_mean(logits_fake-logits_real)
        
    
    def _generetor_loss(self, logits_fake, feature_fake, feature_real):
        self.generator_loss = tf.reduce_mean(-logits_fake)
        
    def train_model(self, dataLoader, max_epoch):
        self.global_steps = 0
        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_value[0], self.clip_value[1])) for
                                         var in self.discriminator_variables]

        for i in range(max_epoch):
            dataLoader.shuffle() 
            batch_gen = dataLoader.batch_generator(batch_size=self.batch_size)

            for batch_imgs in batch_gen:
                batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                feed_dict = {self.z_vec: batch_z, self.real_imgs: batch_imgs, self.train_phase: True}
                
                if self.global_steps < 25 or self.global_steps % 500 == 0:
                    iteration = 25
                else:
                    iteration = self.iter_ratio

                for it_r in range(iteration):
                    self.sess.run(self.discriminator_train_op, feed_dict=feed_dict)
                    self.sess.run(clip_discriminator_var_op)
                    
                self.sess.run(self.generator_train_op, feed_dict=feed_dict)
    
                if self.global_steps % 20 == 0 and self.global_steps != 0:
                    g_loss_val, d_loss_val = self.sess.run(
                        [self.generator_loss, self.discriminator_loss], feed_dict=feed_dict)
                    print("Epoch %d, Step: %d, generator loss: %g, discriminator_loss: %g" % (i, self.global_steps, g_loss_val, d_loss_val))
                    
                self.global_steps += 1
            
            if i % 4 == 0 and i != 0:
                self.saver.save(self.sess, "model/wmodel_%d.ckpt" % self.global_steps, global_step=self.global_steps)
                self._visualize_model('wresult')    
            
