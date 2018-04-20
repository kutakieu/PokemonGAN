import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
# import h5py
# from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
from os import listdir
from os.path import isfile, join
from PIL import Image

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--rnn_hidden', type=int, default=200,
					   help='Number of nodes in the rnn hidden layer')

	parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise dimension')

	parser.add_argument('--word_dim', type=int, default=256,
					   help='Word embedding matrix dimension')

	parser.add_argument('--t_dim', type=int, default=256,
					   help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=20,
					   help='Batch Size')

	parser.add_argument('--image_size', type=int, default=32,
					   help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=64,
					   help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=64,
					   help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024,
					   help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=20,
					   help='Caption Vector Length')

	parser.add_argument('--data_dir', type=str, default="../data",
					   help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
					   help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
					   help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=600,
					   help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=30,
					   help='Save Model/Samples every x iterations over batches')

	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--data_set', type=str, default="flowers",
                       help='Dat set: MS-COCO, flowers')

	args = parser.parse_args()
	model_options = {
		'rnn_hidden' : args.rnn_hidden,
		'word_dim' : args.word_dim,
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length
	}


	gan = model.GAN(model_options)
	with tf.variable_scope(tf.get_variable_scope()) as scope:
		input_tensors, variables, loss, outputs, checks = gan.build_model()



	# with tf.variable_scope("Adam", reuse=False):
	# d_optim = tf.train.AdamOptimizer().minimize(loss['d_loss'], var_list=variables['d_vars'], name='Adam_d')
	# g_optim = tf.train.AdamOptimizer().minimize(loss['g_loss'], var_list=variables['g_vars'], name='Adam_g')
	d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'], name='Adam_d')
	g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'], name='Adam_g')

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)

	loaded_data = load_training_data(args.data_dir, args.data_set, args.caption_vector_length, args.image_size)

	for i in range(args.epochs):
		print("epoch" + str(i))
		batch_no = 0
		index4shuffle = [i for i in range(len(loaded_data))]
		random.shuffle(index4shuffle)

		while (batch_no+1)*args.batch_size < len(loaded_data):
			real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size,
				args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, index4shuffle[batch_no*args.batch_size:(batch_no+1)*args.batch_size], loaded_data)
			# print(caption_vectors)

			# DISCR UPDATE
			check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
			_, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})

			# print("d1", d1)
			# print("d2", d2)
			# print("d3", d3)
			# print("D", d_loss)

			# GEN UPDATE
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})

			# GEN UPDATE TWICE, to make sure d_loss does not go to 0
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})

			# print("LOSSES")
			print("d_loss=" + str(d_loss) + ", g_loss=" + str(g_loss))
			batch_no += 1
			if (batch_no % args.save_every) == 0:
				print("Saving Images, Model")
				save_for_vis(args.data_dir, real_images, gen, image_files)
				save_path = saver.save(sess, "../models/latest_model_{}_temp.ckpt".format(args.data_set))
		print("d1", d1)
		print("d2", d2)
		print("d3", d3)
		print("D", d_loss)
		if i%5 == 0:
			print("===== epoch " + str(i) + " =====")
			save_path = saver.save(sess, "../models/model_after_{}_epoch_{}.ckpt".format(args.data_set, i))

def load_training_data(data_dir, data_set, caption_vector_length, image_size):

	path2training_data = "../data/pokemon_img/"
	filenames = [f for f in listdir(path2training_data) if isfile(join(path2training_data, f))]
	images = []
	pokemon_names = []
	data = []
	for filename in filenames:
		if filename.split(".")[1] == "jpg":
			# filenames.append(filename)
			current_data = {}
			# pokemon_names.append(filename.split(".")[0])
			name_numerical = np.zeros((caption_vector_length))
			current_name = filename.split(".")[0]
			for i, c in enumerate(current_name):
				c = c.lower()
				name_numerical[i] = int(ord(c))
				# if name_numerical[i] == 769:
				# 	print(filename)
				# Flabébé

			current_data["name"] = filename.split(".")[0]
			current_data["name_numerical"] = name_numerical
			# images.append(Image.open(path2training_data + filename).resize(args.image_size))
			current_data["image"] = np.asarray(Image.open(path2training_data + filename).resize([image_size,image_size]))
			data.append(current_data)

	return data


	if data_set == 'flowers':
		h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
		flower_captions = {}
		for ds in h.items():
			flower_captions[ds[0]] = np.array(ds[1])
		image_list = [key for key in flower_captions]
		image_list.sort()

		img_75 = int(len(image_list)*0.75)
		training_image_list = image_list[0:img_75]
		random.shuffle(training_image_list)

		return {
			'image_list' : training_image_list,
			'captions' : flower_captions,
			'data_length' : len(training_image_list)
		}

	else:
		with open(join(data_dir, 'meta_train.pkl')) as f:
			meta_data = pickle.load(f)
		# No preloading for MS-COCO
		return meta_data

def save_for_vis(data_dir, real_images, generated_images, image_files):

	shutil.rmtree( join(data_dir, 'samples') )
	os.makedirs( join(data_dir, 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim,
	caption_vector_length, split, data_dir, data_set, index4shuffle, loaded_data = None):

	# path2training_data = "../data/training/"

	real_images = np.zeros((batch_size, image_size, image_size, 3))
	wrong_images = np.zeros((batch_size, image_size, image_size, 3))
	captions = np.zeros((batch_size, caption_vector_length))
	image_files = []
	# wrong_captions = np.zeros((batch_size, caption_vector_length))

	for i in range(batch_size):
		current_data = loaded_data[index4shuffle[i]]
		real_images[i] = current_data["image"]
		random_index = random.randint(0,index4shuffle[i]-1) if index4shuffle[i]>int(len(loaded_data)/2) else random.randint(index4shuffle[i]+1, len(loaded_data)-1)
		wrong_images[i] = loaded_data[random_index]["image"]
		captions[i] = current_data["name_numerical"]
		image_files.append(current_data["name"])

	z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
	return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
	main()
