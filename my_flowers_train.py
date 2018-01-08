#! /usr/bin/env python2
# -*- coding: utf-8 -*-

'''
训练 flowers: 5 categories (daisy, dandelion, roses, sunflowers, tulips)
数据集用create_tfrecords目录下面的方法创建的tfrecord数据集

构造一个卷积神经网络来训练5 flowers数据集：
图片预处理为： 224x224x3
输入层： 223x223x3
卷积层1: kernel 7x7 stride 4 featuremaps 64  --> [-1, 55, 55, 64]
池化1：  kernel 2x2 stride 1                 --> [-1, 28, 28, 64]
卷积层2: kernel 5x5 stride 2 featuremaps 96  --> [-1, 12, 12, 96]
池化2：  kernel 2x2 stride 1                 --> [-1, 6, 6, 96]
卷积层3: kernel 3x3 stride 2 featuremaps 128  --> [-1, 3, 3, 128]
输出层： 5个输出节点

测试结果：accuracy    loss
2200次：   0.620      1.279
'''

import tensorflow as tf
import argparse
import numpy as np
import os
import struct
import time
from tensorflow.python import debug as tf_debug
slim = tf.contrib.slim
import inception_preprocessing
import random
from PIL import Image
import math

data_path = './flowers_dataset/'
IMAGE_WIDTH = 223
IMAGE_HEIGHT = 223
IAMGE_DETH = 3
NUM_CLASSES = 5
INIT_LEARN_RATE = 0.1

"""清空某个目录下的所有文件"""
def dirClear(path):
	if os.path.isdir(path):
		for file_name in os.listdir(path):
			os.remove(os.path.join(path, file_name))

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor. """
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var) #数学期望
		tf.summary.scalar('mean/'+ name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean))) #方差
		tf.summary.scalar('sttdev/' + name, stddev)
		tf.summary.scalar('max/' + name, tf.reduce_max(var)) #最大值
		tf.summary.scalar('min/' + name, tf.reduce_min(var)) #最小值
		tf.summary.histogram(name, var) #画直方图

#定义初始化操作
def weight_variable(shape, name = None, std = 0.1):
	init = tf.truncated_normal(shape, stddev = std)
	return tf.Variable(init, name = name)

def bias_variable(shape, name = None):
	init = tf.constant(0.1, shape = shape)
	return tf.Variable(init, name = name)

#定义卷积和池化操作
'''
卷积后的图像高宽计算公式： W2 = (W1 - Fw + 2P) / S + 1
其中：Fw为filter的宽，P为周围补0的圈数，S是步幅
'''
def conv2d(x, W, stride = 1, padding = 'SAME'):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def max_pool_2x2(x, padding = 'SAME'):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

'''
(1) 数据加载
'''
#Create a dictionary to refer each label to their string name
labels_file = './flowers_dataset/labels.txt'
labels_to_name = {}
with open(labels_file, 'r') as labels:
	for line in labels:
		label, string_name = line.split(':')
		string_name = string_name[:-1] #Remove newline
		labels_to_name[int(label)] = string_name
		print(label+' : '+string_name)

#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}

#=========================================== 图片旋转===================================
#先放大再旋转再裁剪为原图大小: 输入为np数组
def rotate_image_inflate_np(src_image):
	#判断是否是Numpy数组
	if not isinstance(src_image, np.ndarray):
		raise ValueError('src_image should be a numpy array!!!!')
	angle = random.randint(-20, 20) #随机参数 -20 ~ 20度
	print('random angle = %d' % (angle))

	if angle == 0:
		return src_image

	#print('@inflate then rotate image: %d degree' % (angle))
	image = Image.fromarray(src_image)
	w1 = image.size[0]
	h1 = image.size[1]
	#print('w1 = %d, h1 = %d' % (w1, h1))

	#计算需要放大后的大小
	w2 = w1 * math.cos(math.radians(abs(angle))) + h1 * math.sin(math.radians(abs(angle)))
	h2 = w1 * math.sin(math.radians(abs(angle))) + h1 * math.cos(math.radians(abs(angle)))
	#print('w2 = %f, h2 = %f' % (w2, h2))
	w2 = int(w2) + 1
	h2 = int(h2) + 1

	#放大图像到 w2 x h2
	#image = image.thumbnail((w2, h2))
	image = image.resize((w2, h2))

	#image = transform.rotate(image, angle, resize = True) #旋转同时改变大小
	image = image.rotate(angle, expand = True)
	#print(image.size)
	w3 = image.size[0]
	h3 = image.size[1]

	#裁剪源图像大小
	left_x = (w3 - w1) / 2
	left_y = (h3 - h1) / 2
	right_x = left_x + w1
	right_y = left_y + h1
	image = image.crop((left_x, left_y, right_x, right_y))
	#print(image.size, type(image))

	return np.asarray(image)

#只支持旋转范围在-20~ 20度之间
def random_rotate_image(src_image):
	src_image = tf.cast(src_image, tf.uint8)
	dst_image = tf.py_func(rotate_image_inflate_np, [src_image], src_image.dtype)
	dst_image = tf.cast(dst_image, tf.float32)
	return dst_image

#=====================================数据预处理=================================
def image_preprocess(image, height, width, is_train = True):
	print('(1)image:', image.shape, image.dtype)
	if image.dtype != tf.float32:
		#转换Image数据为float32类型
		#floatimage = tf.image.convert_image_dtype(image, dtype=tf.float32)
		floatimage = tf.cast(image, tf.float32)
	else:
		floatimage = image
	print('(2)floatimage:', floatimage.shape, floatimage.dtype)
	
	#裁剪为width * height大小的图像
	if is_train == False:
		print('(3)floatimage:', floatimage.shape, floatimage.dtype)
		#floatimage_resize = tf.image.resize_image_with_crop_or_pad(floatimage, height, width)
		floatimage_resize = tf.image.central_crop(floatimage, central_fraction=0.875)
		floatimage_resize = tf.expand_dims(floatimage_resize, 0)
		floatimage_resize = tf.image.resize_bilinear(floatimage_resize, [height, width], align_corners=False)
		floatimage_resize = tf.squeeze(floatimage_resize, [0])
	else:
		#传到这一步的数据一定要是float类型，如果是uint8，经过测试Loss下降不了
		#floatimage_resize = tf.random_crop(floatimage, [height, width, 3]) #随机裁剪
		floatimage_resize = tf.image.resize_image_with_crop_or_pad(floatimage, height, width)
		#随机旋转-20 ~ 20度
		floatimage_resize = random_rotate_image(floatimage_resize)
		print('-----', type(floatimage_resize), floatimage_resize.dtype, floatimage_resize.shape)
		floatimage_resize = tf.reshape(floatimage_resize, [height, width, 3])
		print('(3)floatimage_resize:', floatimage_resize.shape, floatimage_resize.dtype)
		floatimage_resize = tf.image.random_flip_left_right(floatimage_resize) #随机左右翻转
		floatimage_resize = tf.image.random_brightness(floatimage_resize, max_delta=63) #随机调整亮度
		floatimage_resize = tf.image.random_contrast(floatimage_resize, lower=0.2, upper=1.8) #随机调整对比度
	print('(4)floatimage_resize:', floatimage_resize.shape, floatimage_resize.dtype)

	floatimage_resize = tf.image.per_image_standardization(floatimage_resize)
	return floatimage_resize

#=====================================flowers tfrecord 数据加载======================================
def get_split(split_name, dataset_dir, file_pattern, file_pattern_for_counting='flowers'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. 用于区分训练数据文件和测试文件
    - dataset_dir(str): 数据集存放目录
    - file_pattern(str): 数据集文件名字的匹配模式
    - file_pattern_for_counting(str): 用于统计训练集合测试集各有多少个文件的匹配字符串

    OUTPUTS:
    - dataset (Dataset): 返回数据集类型对象
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
		'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
		'image/class/label': tf.FixedLenFeature(
			[], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
		'image': slim.tfexample_decoder.Image(),
		'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = NUM_CLASSES,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset, num_samples

def load_batch(dataset, batch_size, num_samples, height, width, is_training=True):
	min_queue_samples = int(num_samples * 0.4)
	#First create the data_provider object
	data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
		common_queue_capacity = min_queue_samples + 3 * batch_size,
		common_queue_min = min_queue_samples)
		
	#Obtain the raw image using the get method
	raw_image, label = data_provider.get(['image', 'label'])
	
	print('raw_image:', raw_image.shape, raw_image.dtype)
	#Perform the correct preprocessing for this image depending if it is training or evaluating
	#image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
	image = image_preprocess(raw_image, height, width, is_training)
	print('image:', image.shape, image.dtype)
	
	#As for the raw images, we just do a simple reshape to batch it up
	raw_image = tf.expand_dims(raw_image, 0)
	raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
	raw_image = tf.squeeze(raw_image)
	
	#Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
	if not is_training:
		images, labels = tf.train.batch(
			[image, label],
			batch_size = batch_size,
			num_threads = 16,
			capacity = min_queue_samples + 3 * batch_size)
		tf.summary.image('inputImages', images, batch_size) #测试时查看图片数据
	else:
		images, labels = tf.train.shuffle_batch(
			[image, label],
			batch_size = batch_size,
			num_threads = 16,
			capacity = min_queue_samples + 3 * batch_size,
			min_after_dequeue = min_queue_samples)
		tf.summary.image('inputImages', images, 10) #训练时查看图片数据
	
	return images, labels

#main函数入口
def main(_):
	model_save_path = './model_data/'
	train_logs_path = './log/'
	'''
	(2) 输入层，输入张量x定义
	'''
	#神经网络输入层变量x定义
	with tf.name_scope('input_layer'):
		split_name = 'validation' if ARGS.test else 'train'
		print 'split_name: ' + split_name
		dataset, num_samples = get_split(split_name, data_path, 'flowers_%s_*.tfrecord')
		print 'num_samples: %d' % (num_samples)
		images, labels = load_batch(dataset, 128, num_samples, 
							IMAGE_HEIGHT, IMAGE_WIDTH, False if ARGS.test else True)
		#labels to one_hot type
		labels = slim.one_hot_encoding(labels, dataset.num_classes)
		print('images:', images.shape, images.dtype)
		print('labels:', labels.shape, labels.dtype)
		#images = tf.reshape(images, [-1, 223, 223, 3]) # 223x223的RGB彩色图

	'''
	(3) 第一层卷积层
	卷积层1: kernel 7x7 stride 4 featuremaps 64  --> [-1, 55, 55, 64]
	池化1：  kernel 2x2 stride 1                 --> [-1, 28, 28, 64]
	'''
	with tf.name_scope('conv_layer_1'):
		W_conv1 = weight_variable([7, 7, 3, 64], 'Filter1', 1e-4)
		b_conv1 = bias_variable([64])
		variable_summaries(W_conv1, 'W_conv1')
		variable_summaries(b_conv1, 'b_conv1')
		#卷积操作
		'''
		用公式W2 = (W1 - Fw + 2P) / S + 1，55 = (223 - 7 + 2 * 0) / 4 + 1
		'''
		h_conv1 = tf.nn.relu(conv2d(images, W_conv1, 3, 'VALID') + b_conv1)
		#将卷积完的结果进行pooling操作
		#输出h_pool1维度为：[-1, 28, 28, 64]
		h_pool1 = max_pool_2x2(h_conv1)
		h_lrn1 = tf.nn.local_response_normalization(h_pool1)

	'''
	(4) 第二层卷积层
	卷积层2: kernel 5x5 stride 2 featuremaps 96  --> [-1, 12, 12, 96]
	池化2：  kernel 2x2 stride 1                 --> [-1, 6, 6, 96]
	'''
	with tf.name_scope('conv_layer_2'):
		W_conv2 = weight_variable([5, 5, 64, 96], 'Filter2', 1e-4)
		b_conv2 = bias_variable([96])
		variable_summaries(W_conv2, 'W_conv2')
		variable_summaries(b_conv2, 'b_conv2')
		#卷积操作
		#用公式W2 = (W1 - Fw + 2P) / S + 1，12 = (28 - 5 + 2 * 0) / 2 + 1
		h_conv2 = tf.nn.relu(conv2d(h_lrn1, W_conv2, 2, 'VALID') + b_conv2)
		#将卷积完的结果进行pooling操作
		#输出h_pool2维度为：[-1, 6, 6, 96]
		h_pool2 = max_pool_2x2(h_conv2)

	'''
	(5) 第三层卷积层
	卷积层3: kernel 3x3 stride 2 featuremaps 128  --> [-1, 3, 3, 128]
	'''
	with tf.name_scope('conv_layer_3'):
		#定义卷积操作的map为5x5的矩阵，且输出64个feature map, 输入的图片的通道数为32
		W_conv3 = weight_variable([3, 3, 96, 128], 'Filter3', 1e-4)
		b_conv3 = bias_variable([128])
		variable_summaries(W_conv3, 'W_conv3')
		variable_summaries(b_conv3, 'b_conv3')
		#卷积操作
		#输出h_conv3维度为：[-1, 3, 3, 128]
		h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 2) + b_conv3)

	'''
	(5) 全连接层定义（full connection layer）
	'''
	with tf.name_scope('full_connection_layer'):
		#卷积层2输出作为输入和隐藏层之间的权重矩阵W_fc1,偏置项b_fc1初始化
		'''(1) 输入层 -》 隐藏层1'''
		with tf.name_scope('connect1'):
			#定义隐藏层的节点数
			hide1_neurons = 1024
			#计算卷积层2输出的tensor，变化为一维的大小
			h_conv3_shape = h_conv3.get_shape().as_list() #得到一个列表[batch, hight, width, channels]
			fc_input_size = h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3] # hight * width * channels
			W_fc1 = weight_variable([fc_input_size, hide1_neurons], 'W_fc1')
			b_fc1 = bias_variable([hide1_neurons], 'b_fc1')
			variable_summaries(W_fc1, 'W_fc1')
			variable_summaries(b_fc1, 'b_fc1')
			#将卷积层2的输出张量扁平化作为全连接神经网络的输入
			fc_x = tf.reshape(h_conv3, [-1, fc_input_size], name = 'fc_x')
			#全连接层中隐藏层1的输出
			fc_h1 = tf.nn.relu(tf.matmul(fc_x, W_fc1) + b_fc1, name = 'fc_h1')
		
		'''(2) 隐藏层1 -》隐藏层2'''
		with tf.name_scope('connect2'):
			#定义隐藏层的节点数
			hide2_neurons = 1024
			W_fc2 = weight_variable([hide1_neurons, hide2_neurons], 'W_fc2')
			b_fc2 = bias_variable([hide2_neurons], 'b_fc2')
			variable_summaries(W_fc2, 'W_fc2')
			variable_summaries(b_fc2, 'b_fc2')
			#全连接层中隐藏层2的输出
			fc_h2 = tf.nn.relu(tf.matmul(fc_h1, W_fc2) + b_fc2, name = 'fc_h2')

		'''(3) 隐藏层2 -》输出层'''
		with tf.name_scope('connect3'):
			#为了减少过拟合，在隐藏层和输出层之间加人dropout操作。
			#用来代表一个神经元的输出在dropout中保存不变的概率。
			#在训练的过程启动dropout，在测试过程中关闭dropout
			keep_prob = tf.placeholder("float")
			drop_fc_h = tf.nn.dropout(fc_h2, keep_prob)

			#隐藏层到输出层
			W_fc3 = weight_variable([hide2_neurons, NUM_CLASSES], 'W_fc3')
			b_fc3 = bias_variable([NUM_CLASSES], 'b_fc3')
			variable_summaries(W_fc3, 'W_fc3')
			variable_summaries(b_fc3, 'b_fc3')
			y = tf.nn.softmax(tf.matmul(drop_fc_h, W_fc3) + b_fc3, name = 'y')

	'''
	(6) 设置训练方法，及其他超参数
	'''
	with tf.name_scope('loss_mean'):
		#设置期待输出值
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
						logits = y, labels = labels, name = 'cross_entropy_per_example')
		loss_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
		tf.summary.scalar('entropy_loss_mean', loss_mean)

	with tf.name_scope('train_step'):
		#使用更复杂的ADAM优化器来做梯度最速下降
		optimizer = tf.train.AdamOptimizer(1e-4)
		train_step = optimizer.minimize(cross_entropy)

	with tf.name_scope('init'):
		#初始化
		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()

	with tf.name_scope('saver'):
		#定义saver用来保存训练好的模型参数
		saver = tf.train.Saver()

	#定义检测正确率的方法
	with tf.name_scope('accuracy'):
		actual_index = tf.argmax(labels, 1)
		predict_index = tf.argmax(y, 1)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1)) #用向量y和y_中的最大值进行比较
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #对正确率求均值
		tf.summary.scalar('accuracy', accuracy)

	summary = tf.summary.merge_all()

	def train_and_test():
		#建立会话
		#with tf.Session() as sess:
		sess = tf.Session()
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		'''
		(7) 开始训练
		'''
		#执行初始化
		sess.run(init)

		train_writer = tf.summary.FileWriter(train_logs_path, sess.graph)
		
		start_time = time.time() #记录开始时间

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess, coord = coord)

		#用于统计各个节点的运行时间
		train_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
		train_metadata = tf.RunMetadata()

		''' 用最近检查点保存的数据，继续训练模型 '''
		model_data = None
		model_data = tf.train.latest_checkpoint(model_save_path)
		#saver.restore(sess, './save_model_data_conv_add_blur/model_iter.ckpt-110000')
		if model_data != None:
			saver.restore(sess, model_data)
		print('restore: %s' % model_data)

		#开始训练200000次
		for i in range(200000):
			sess.run(train_step, feed_dict={keep_prob: 0.5})

			#每100次训练完都检测下测试的正确率
			if i % 10 == 0:
				cur_rate, cur_loss, summary_train = sess.run([accuracy, loss_mean, summary],
															feed_dict = {keep_prob: 1},
															options = train_options, 
															run_metadata = train_metadata)
				print('epoch %d accuracy: %s, loss: %s' % (i, cur_rate, cur_loss))

			if i % 100 == 0:
				#assert not np.isnan(cur_loss), 'Model diverged with loss = NaN'
				# Update the events file.
				train_writer.add_summary(summary_train, i)
				train_writer.add_run_metadata(train_metadata, 'step%d' % i)
				train_writer.flush()
			
			if i % 100 == 0:
				model_data_file = os.path.join(model_save_path, 'model_iter.ckpt')
				print('#######mode save file: %s' % model_data_file)
				saver.save(sess, model_data_file, global_step = i)

		duration = time.time() - start_time
		print('------Train network spend: %.3f seconds-------' % duration)

		#保存训练后的模型参数
		model_data_file = os.path.join(model_save_path, 'model.ckpt')
		print('########mode save file: %s' % model_data_file)
		saver.save(sess, model_data_file)

		coord.request_stop()
		coord.join(threads)
		sess.close()

	#测试训练完的模型正确率
	def test_model_accuracy():
		dirClear('./test_log/')
		with tf.Session() as sess:
			#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			''' 恢复训练好的数据 '''
			model_data = tf.train.latest_checkpoint('./model_data/')
			if model_data == None:
				raise ValueError('model_data is None')
			saver.restore(sess, model_data)
			print('restore file: %s' % model_data)

			test_writer = tf.summary.FileWriter('./test_log/', sess.graph)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess, coord = coord)

			#只好多次加载统计结果
			sum_rate = 0
			sum_loss = 0
			n = num_samples // 128 + 1
			print('n = %d' % (n))
			for i in range(n):
				epoch_rate, epoch_loss, summary_test, actual_index_test, predict_index_test = sess.run(
					[accuracy, loss_mean, summary, actual_index, predict_index], 
					feed_dict = {keep_prob: 1})
				print('[%d]epoch_rate: %s, epoch_loss: %s' % (i, epoch_rate, epoch_loss))
				
				print '===================================================='
				j = 0
				print('actual_index_test:')
				for actualIndex, predictIndex in zip(actual_index_test, predict_index_test):
					if actualIndex != predictIndex:
						print('\033[1;31;43mPicture %d actual is %s, predict is %s\033[0m' % 
								(j, labels_to_name[actualIndex], labels_to_name[predictIndex]))
					else:
						print('Picture %d actual is %s, predict is %s' % 
							(j, labels_to_name[actualIndex], labels_to_name[predictIndex]))
					j += 1
				print '===================================================='

				#print(h_pool2.shape)
				sum_rate = sum_rate + epoch_rate
				sum_loss = sum_loss + epoch_loss

				test_writer.add_summary(summary_test, i)
				test_writer.flush()
			print('The Accuracy tested by flowers-5 Test samples is: %s, loss is: %s' % ((sum_rate / n), (sum_loss / n)))
			coord.request_stop()
			coord.join(threads)


	if ARGS.test:
		print('************** Predict by Neural network ***************')
		test_model_accuracy()
	else:
		print('************** Train and Test accuracy of Neural network ***************')
		#dirClear(train_logs_path)
		train_and_test()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-t',
		'--test',
		default = False,
		action = 'store_true',
		help = 'Excute training network or testing network.'
		)
  
	ARGS = parser.parse_args()
	print('ARGS: %s' % ARGS)
	tf.app.run()

'''
(1)tf.argmax(input, axis=None, name=None, dimension=None)
此函数是对矩阵按行或列计算最大值
参数

    input：输入Tensor
    axis：0表示按列，1表示按行
    name：名称
    dimension：和axis功能一样，默认axis取值优先。新加的字段

返回：Tensor 行或列的最大值下标向量

(2)tf.equal(a, b)
此函数比较等维度的a, b矩阵相应位置的元素是否相等，相等返回True,否则为False
返回：同维度的矩阵，元素值为True或False

(3)tf.cast(x, dtype, name=None)
将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以

(4)tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
 功能：求某维度的最大值
(5)tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
功能：求某维度的均值

参数1--input_tensor:待求值的tensor。
参数2--reduction_indices:在哪一维上求解。0表示按列，1表示按行
参数（3）（4）可忽略
例：x = [ 1, 2
		  3, 4]
x = tf.constant([[1,2],[3,4]], "float")
tf.reduce_mean(x) = 2.5
tf.reduce_mean(x, 0) = [2, 3]
tf.reduce_mean(x, 1) = [1.5, 3.5]

(6)tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
从截断的正态分布中输出随机值

    shape: 输出的张量的维度尺寸。
    mean: 正态分布的均值。
    stddev: 正态分布的标准差。
    dtype: 输出的类型。
    seed: 一个整数，当设置之后，每次生成的随机数都一样。
    name: 操作的名字。

（7）tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
从标准正态分布中输出随机值

(8) tf.nn.conv2d(input, filter, strides, padding, 
				 use_cudnn_on_gpu=None, data_format=None, name=None)
在给定的4D input与 filter下计算2D卷积
	1，输入shape为 [batch, height, width, in_channels]: batch为图片数量，in_channels为图片通道数
	2，第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, 
		in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，
		卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input
		的第四维
	3，第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
	4，第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
	5，第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true

	结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。

(9)tf.nn.max_pool(value, ksize, strides, padding, name=None)
参数是四个，和卷积很类似：
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
	依然是[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们
	不想在batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式

(10) tf.reshape(tensor, shape, name=None)
函数的作用是将tensor变换为参数shape的形式。
其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小，
函数会自动计算，但列表中只能存在一个-1。（当然如果存在多个-1，就是一个存在多解的方程了）

(11)tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None) 
为了减少过拟合，随机扔掉一些神经元，这些神经元不参与权重的更新和运算
参数：
	x            :  输入tensor
	keep_prob    :  float类型，每个元素被保留下来的概率
	noise_shape  : 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。
	seed         : 整形变量，随机数种子。
	name         : 名字，没啥用。 
'''