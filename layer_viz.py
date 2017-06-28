import sys
import pprint
import math
from utils import Data
import model
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
def plotFeatureMaps(units):
	left  = 0.125  # the left side of the subplots of the figure
	right = 0.9    # the right side of the subplots of the figure
	bottom = 0.1   # the bottom of the subplots of the figure
	top = 0.5      # the top of the subplots of the figure
	wspace = 0.1   # the amount of width reserved for blank space between subplots
	hspace = 1   # the amount of height reserved for white space between subplots

	filters = units.shape[3]
	plt.figure(1, figsize=(20,6))
	# plt.figure(1, figsize=(20,8))
	n_columns = 6
	n_rows = math.ceil(filters / n_columns) + 1
	for i in range(filters):
		plt.subplot(n_rows, n_columns, i+1)
		plt.gca().get_xaxis().set_visible(False)
		plt.gca().get_yaxis().set_visible(False)
		# plt.title('Map ' + str(i))
		plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
	# plt.suptitle('Feature Maps - Third Conv Layer', fontsize=20)
	# plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	# plt.subplots_adjust(wspace=0.02, hspace=0.)
	plt.subplots_adjust(wspace=0.02, hspace=0.02)
	plt.show()
	# plt.savefig('maps.png')


# visualize first layer filters
def show_filters():
	# only first layer filters make sense (according to Karpathy - cs231n)
	layer = model.W_conv1
	print model.flattened_length
	# n_filters = int(layer.get_shape().as_list()[3])
	# tmp = layer
	# tensorFilters = tf.split(tmp, n_filters, 3) # 24 x (5, 5, 3, 1) list of 24 tensors
	# plt.figure(1, figsize=(12,9))
	# n_columns = 6
	# n_rows = n_filters / n_columns
	# for i in range(n_filters):
	# 	im = sess.run(tensorFilters[i])
	# 	im = im.reshape((5,5,3))
	# 	plt.subplot(n_rows, n_columns, i+1)
	# 	plt.gca().get_xaxis().set_visible(False)
	# 	plt.gca().get_yaxis().set_visible(False)
	# 	plt.title('Kernel ' + str(i))
	# 	plt.imshow(im)
	# plt.suptitle('Kernels visualization', fontsize=20)
	# plt.show()
	# plt.savefig('kernels_visualization.png')


# visualize feature maps
def show_maps():
	layer = model.h_conv
	data = Data(1)
	# print data._y.shape
	# print data._X.shape
	# layer = model.h_conv5
	batch_size = 40  # image that will be used, taken from test_data/X.npy
	batch = data.next_batch(batch_size) # jump batch_size images
	batch = data.next_batch(batch_size) # use the first image of this batch
	feature_maps = sess.run(layer,feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
	plotFeatureMaps(feature_maps)


if __name__ == '__main__':
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	saver.restore(sess, "./model.ckpt")

	if sys.argv[1] == 'filters': # python layer_viz.py filters
		show_filters()
	elif sys.argv[1] == 'maps': # python layer_viz.py maps
		show_maps()

