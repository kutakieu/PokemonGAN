
# -*- coding: utf-8 -*-

# generate new kinds of pokemons

import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *
import argparse
import pickle

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
version = 'newPokemon'
newPoke_path = './' + version

def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)

def process_data():
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, 'pokemon_img')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images
    all_images = tf.convert_to_tensor(images, dtype = tf.string)

    images_queue = tf.train.slice_input_producer([all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise'))
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    # iamges_batch = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE, num_threads=4, capacity = 200 + 3*BATCH_SIZE, min_after_dequeue=200)
    iamges_batch = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE, num_threads=1, capacity = len(images)*10+ 3*BATCH_SIZE, min_after_dequeue=len(images))
    num_images = len(images)

    return iamges_batch, num_images

def rnn(input_, output_size, embedding_size, n_hidden, stddev=0.02, bias_start=0.0, dropout_rate=0.5, reuse=False):

	# with tf.variable_scope("rnn4name", reuse=reuse):
	# with tf.variable_scope("d_g_rnn4name", reuse=tf.AUTO_REUSE):
    with tf.variable_scope('rnn4name') as scope:
        if reuse:
            scope.reuse_variables()
        word_embeddings = tf.get_variable("char_embeddings", [150, embedding_size])
        embedded_word_output = tf.nn.embedding_lookup(word_embeddings, input_)
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=False)
        # lstm_fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden, reuse=False)

        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=False)
        # lstm_bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden, reuse=False)

        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_rate)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_rate)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_word_output, dtype=tf.float32)

        fw_output = outputs[0]
        bw_output = outputs[1]
        lstm_output = tf.concat((fw_output, bw_output), 2)

        matrix = tf.get_variable("Weight", [n_hidden*2, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        return tf.matmul(lstm_output[:,-1,:], matrix) + bias

def cnn_rnn(input_, output_size, embedding_size, n_hidden, stddev=0.02, bias_start=0.0, dropout_rate=0.5, reuse=False):

    with tf.variable_scope('cnn_rnn') as scope:
        if reuse:
            scope.reuse_variables()

        word_embeddings = tf.get_variable("char_embeddings", [150, embedding_size])
        embedded_word_output = tf.nn.embedding_lookup(word_embeddings, input_)

        conv1 = tf.layers.conv1d(embedded_word_output, 384, 4, strides=1)
        act1 = lrelu(conv1, n='act1')
        # pooled1 = tf.layers.max_pooling1d(act1, pool_size=2, strides=2)

        conv2 = tf.layers.conv1d(act1, 512, 4, strides=1)
        act2 = lrelu(conv2, n='act2')
        # pooled2 = tf.layers.max_pooling1d(act2, pool_size=2, strides=2)

        conv3 = tf.layers.conv1d(act2, 256, 4, strides=1)
        act3 = lrelu(conv3, n='act3')
        # pooled3 = tf.layers.max_pooling1d(act3, pool_size=2, strides=2)


        # word_embeddings = tf.get_variable("char_embeddings", [150, embedding_size])
        # embedded_word_output = tf.nn.embedding_lookup(word_embeddings, input_)

        # Forward direction cell
        # lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=False)
        lstm_fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden)

        # Backward direction cell
        # lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=False)
        lstm_bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden)

        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_rate)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_rate)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, act3, dtype=tf.float32)

        fw_output = outputs[0]
        bw_output = outputs[1]
        lstm_output = tf.concat((fw_output, bw_output), 2)

        matrix = tf.get_variable("Weight", [n_hidden*2, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        # return tf.matmul(lstm_output[:,-1,:], matrix) + bias
        return tf.matmul(tf.reduce_mean(lstm_output, axis=1), matrix) + bias


def generator(input, random_dim, t_caption, is_train, reuse=False):

    # reduced_text_embedding = rnn(t_caption, args.t_dim, args.word_dim, args.rnn_hidden)
    reduced_text_embedding = cnn_rnn(t_caption, args.t_dim, args.word_dim, args.rnn_hidden)
    input = tf.concat([input, reduced_text_embedding], axis=1)

    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        #Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')

        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, t_caption, is_train, reuse=False, t_text_embedding=None):
    # reduced_text_embedding = rnn(t_caption, args.t_dim, args.word_dim, args.rnn_hidden, reuse=True)
    reduced_text_embedding = cnn_rnn(t_caption, args.t_dim, args.word_dim, args.rnn_hidden, reuse=True)

    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat!
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
        #Convolution, activation, bias, repeat!
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat!
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat!
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')


        w2 = tf.get_variable('w2', shape=[fc1.shape[-1]+args.t_dim, 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        fc1 = tf.concat([fc1, reduced_text_embedding], axis=1)
        wgan_logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        dcgan_logits = tf.nn.sigmoid(wgan_logits)
        return wgan_logits , dcgan_logits


def train():
    args.z_dim = 100
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, args.z_dim+args.num_attributes], name='rand_input')
        # random_input = tf.placeholder(tf.float32, shape=[None, args.z_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
        t_attributes = tf.placeholder('float32', shape=[None, args.num_attributes])
        t_caption = tf.placeholder('int32', [None, args.caption_vector_length], name='caption_input')

    # wgan
    fake_image = generator(random_input, args.z_dim+args.num_attributes+args.t_dim, t_caption, is_train)
    # fake_image = generator(random_input, args.z_dim, is_train)

    real_result, logits_real = discriminator(real_image, t_caption, is_train)
    fake_result, logits_fake = discriminator(fake_image, t_caption, is_train, reuse=True)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    #TODO:put conditional cost function using caption
    # d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=t_attributes), axis=1)
    # d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=t_attributes), axis=1)

    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    # image_batch, samples_num = process_data()

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, save_path)
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    loaded_data = load_training_data(args.data_dir, args.data_set, args.caption_vector_length, args.image_size)
    batch_num = int(len(loaded_data) / args.batch_size)
    print('total training sample num:%d' % len(loaded_data))
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (args.batch_size, batch_num, EPOCH))
    print('start training...')

    index4shuffle = [i for i in range(len(loaded_data))]

    for i in range(EPOCH):
        print("epoch " + str(i))

        for j in range(batch_num):
            #print(j)
            d_iters = 5
            g_iters = 1
            random.shuffle(index4shuffle)
            batch_no = 0

            train_noise = np.random.uniform(-1.0, 1.0, size=[args.batch_size, args.z_dim]).astype(np.float32)
            for k in range(d_iters):
                #print(k)
                # train_image = sess.run(image_batch)
                real_images, wrong_images, caption_vectors, z_noise, image_files, attributes = get_training_batch(batch_no, args.batch_size,
    				args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, index4shuffle[batch_no*args.batch_size:(batch_no+1)*args.batch_size], args.num_attributes, loaded_data)

                z_concat = np.concatenate([train_noise, attributes], axis=1)
                # z_concat = np.concatenate([z_concat, reduced_text_embedding], axis=1)

                batch_no += 1
                #wgan clip weights
                sess.run(d_clip)

                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: z_concat, real_image: real_images, is_train: True, t_attributes: attributes, t_caption: caption_vectors})

            # Update the generator
            for k in range(g_iters):
                train_noise = np.random.uniform(-1.0, 1.0, size=[args.batch_size, args.z_dim]).astype(np.float32)
                real_images, wrong_images, caption_vectors, z_noise, image_files, attributes = get_training_batch(batch_no, args.batch_size,
    				args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, index4shuffle[batch_no*args.batch_size:(batch_no+1)*args.batch_size], args.num_attributes, loaded_data)
                z_concat = np.concatenate([train_noise, attributes], axis=1)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: z_concat, is_train: True, t_caption: caption_vectors})

            # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)

        # save check point every 500 epoch
        if i%500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))
        if i%50 == 0:
            # save images
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[args.batch_size, args.z_dim]).astype(np.float32)
            real_images, wrong_images, caption_vectors, z_noise, image_files, attributes = get_training_batch(batch_no, args.batch_size,
                args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, index4shuffle[batch_no*args.batch_size:(batch_no+1)*args.batch_size], args.num_attributes, loaded_data)
            z_concat = np.concatenate([sample_noise, attributes], axis=1)
            # imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            imgtest = sess.run(fake_image, feed_dict={random_input: z_concat, is_train: False, t_attributes: attributes, t_caption: caption_vectors})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')
            print(attributes)

            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    # coord.request_stop()
    # coord.join(threads)


# def test():
    # random_dim = 100
    # with tf.variable_scope('input'):
        # real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        # random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        # is_train = tf.placeholder(tf.bool, name='is_train')

    # # wgan
    # fake_image = generator(random_input, random_dim, is_train)
    # real_result = discriminator(real_image, is_train)
    # fake_result = discriminator(fake_image, is_train, reuse=True)
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    # print(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore)
    # ckpt = tf.train.latest_checkpoint('./model/' + version)
    # saver.restore(sess, ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_hidden', type=int, default=200, help='Number of nodes in the rnn hidden layer')

    parser.add_argument('--z_dim', type=int, default=100, help='Noise dimension')

    parser.add_argument('--num_attributes', type=int, default=18, help='Number of features of each Pokemon; Types, HP, Attack, Defense, SpAttack, SpDefense, Speed')

    parser.add_argument('--word_dim', type=int, default=256, help='Word embedding matrix dimension')

    parser.add_argument('--t_dim', type=int, default=256, help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')

    parser.add_argument('--image_size', type=int, default=128, help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64, help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64, help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024, help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=20, help='Caption Vector Length')

    parser.add_argument('--data_dir', type=str, default="./pokemon_img/", help='Data Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5, help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600, help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=30, help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None, help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="flowers", help='Dat set: MS-COCO, flowers')

    args = parser.parse_args()
    model_options = {
		'rnn_hidden' : args.rnn_hidden,
		'word_dim' : args.word_dim,
		'z_dim' : args.z_dim,
		'num_attributes' : args.num_attributes,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length
	}
    train()
    # test()
