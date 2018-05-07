import tensorflow as tf
# from Utils import ops
import ops
class GAN:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''
	def __init__(self, options):
		self.options = options

		self.g_bn0 = ops.batch_norm(name='g_bn0')
		self.g_bn1 = ops.batch_norm(name='g_bn1')
		self.g_bn2 = ops.batch_norm(name='g_bn2')
		self.g_bn3 = ops.batch_norm(name='g_bn3')

		self.d_bn1 = ops.batch_norm(name='d_bn1')
		self.d_bn2 = ops.batch_norm(name='d_bn2')
		self.d_bn3 = ops.batch_norm(name='d_bn3')
		self.d_bn4 = ops.batch_norm(name='d_bn4')


	def build_model(self):
		img_size = self.options['image_size']
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
		t_real_caption = tf.placeholder('int32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')

		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		t_attributes = tf.placeholder('float32', [self.options['batch_size'], self.options['num_attributes']])

		fake_image = self.generator(t_z, t_attributes)

		disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_attributes)
		disc_wrong_image, disc_wrong_image_logits   = self.discriminator(t_wrong_image, t_attributes, reuse=True)
		disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_attributes, reuse=True)

		#
		# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image)))
		#
		# d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
		# d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits, labels=tf.zeros_like(disc_wrong_image)))
		# d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))

		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=t_attributes))

		d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=t_attributes))
		d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits, labels=t_attributes))
		d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=t_attributes))

		d_loss = d_loss1 + d_loss2 + d_loss3

		t_vars = tf.trainable_variables()
		# print(t_vars)
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z,
			't_attributes' : t_attributes
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'disc_real_image_logits' : disc_real_image_logits,
			'disc_wrong_image_logits' : disc_wrong_image,
			'disc_fake_image_logits' : disc_fake_image_logits
		}

		return input_tensors, variables, loss, outputs, checks

	def build_generator(self):
		img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		fake_image = self.sampler(t_z)

		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}

		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

	# Sample Images for a text embedding
	def sampler(self, t_z, t_attributes=None, t_text_embedding=None):
		tf.get_variable_scope().reuse_variables()

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		""" dim 2400 -> 256 """
		# reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		if t_text_embedding == None:
			if t_attributes == None:
				z_concat = t_z
			else:
				z_concat = tf.concat([t_z, t_attributes], 1)
		else:
			reduced_text_embedding = rnn(t_text_embedding, self.options['t_dim'], self.options['word_dim'], self.options['rnn_hidden'])
			z_concat = tf.concat([t_z, reduced_text_embedding], 1)

		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train = False))

		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train = False))

		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train = False))

		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train = False))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

		return (tf.tanh(h4)/2. + 0.5)

	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_attributes=None, t_text_embedding=None):

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

		# reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		if t_text_embedding == None:
			if t_attributes == None:
				z_concat = t_z
			else:
				z_concat = tf.concat([t_z, t_attributes], 1)
		else:
			reduced_text_embedding = rnn(t_text_embedding, self.options['t_dim'], self.options['word_dim'], self.options['rnn_hidden'])
			z_concat = tf.concat([t_z, reduced_text_embedding], 1)

		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0))

		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1))

		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2))

		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

		return (tf.tanh(h4)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_attributes=None, t_text_embedding=None, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		h1 = ops.lrelu( self.d_bn1(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))) #16
		h2 = ops.lrelu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))) #8
		h3 = ops.lrelu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))) #4

		# ADD TEXT EMBEDDING TO THE NETWORK
		# TODO: replace this part with charcter base RNN
		# reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))

		if t_text_embedding == None:
			if t_attributes == None:
				h3_concat = h3
			else:
				t_attributes = tf.expand_dims(t_attributes,1)
				t_attributes = tf.expand_dims(t_attributes,2)
				tiled_attributes = tf.tile(t_attributes, [1,2,2,1], name='tiled_attributes')
				h3_concat = tf.concat([h3, tiled_attributes], 3, name='h3_concat')
		else:
			reduced_text_embeddings = ops.rnn(t_text_embedding, self.options['t_dim'], self.options['word_dim'], self.options['rnn_hidden'], reuse=True)
			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
			tiled_embeddings = tf.tile(reduced_text_embeddings, [1,2,2,1], name='tiled_embeddings')
			h3_concat = tf.concat([h3, tiled_embeddings], 3, name='h3_concat')

		h3_new = ops.lrelu( self.d_bn4(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4

		# h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), self.options['num_attributes'], 'd_h3_lin')

		return tf.nn.sigmoid(h4), h4
