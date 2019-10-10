import h5py 
import numpy
import tensorflow as tf
import time
import sys
import os
from class_util import classes

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate


BASE_LEARNING_RATE = 2e-4
NUM_FEATURE_CHANNELS = [64,64,128,512]
NUM_CONV_LAYERS = len(NUM_FEATURE_CHANNELS)
NUM_FC_CHANNELS = [512]
NUM_FC_LAYERS = len(NUM_FC_CHANNELS)

BATCH_SIZE = 100
NUM_POINT = 1024
NUM_CLASSES = len(classes)
MAX_EPOCH = 50
VAL_STEP = 10
GPU_INDEX = 0

class PointNet():
	def __init__(self,batch_size,num_point,num_class):
		#inputs
		input_channels = 6
		self.pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, input_channels))
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))	
		self.is_training_pl = tf.placeholder(tf.bool, shape=())
		self.input = tf.expand_dims(self.pointclouds_pl,-1)
		self.conv = [None] * NUM_CONV_LAYERS
		self.kernel = [None] * NUM_CONV_LAYERS
		self.bias = [None] * NUM_CONV_LAYERS
		self.fc = [None] * (NUM_FC_LAYERS + 1)
		self.fc_weights = [None] * (NUM_FC_LAYERS + 1)
		self.fc_bias = [None] * (NUM_FC_LAYERS + 1)
		self.pool = [None]
		self.tile = [None] * 2

		#hidden layers
		for i in range(NUM_CONV_LAYERS):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [1,input_channels if i==0 else 1, 1 if i==0 else NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.nn.conv2d(self.input if i==0 else self.conv[i-1], self.kernel[i], [1, 1, 1, 1], padding='VALID')
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = tf.nn.relu(self.conv[i])

		self.pool[0] = tf.nn.max_pool(self.conv[-1],ksize=[1, num_point, 1, 1],strides=[1, num_point, 1, 1], padding='VALID', name='pool'+str(i))
		self.tile[0] = tf.tile(tf.reshape(self.pool[0],[batch_size,-1,NUM_FEATURE_CHANNELS[-1]]) , [1,1,num_point])
		self.tile[0] = tf.reshape(self.tile[0],[batch_size,num_point,-1])
		self.tile[0] = tf.reshape(self.conv[-1],[batch_size,num_point,-1]) - self.tile[0]
		self.tile[1] = tf.reshape(self.conv[1], [batch_size, num_point, -1])
		self.concat = tf.concat(axis=2, values=self.tile)

		def batch_norm_template(inputs, is_training, moments_dims):
			with tf.variable_scope('bn') as sc:
				num_channels = inputs.get_shape()[-1].value
				beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
					name='beta', trainable=True)
				gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
					name='gamma', trainable=True)
				batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
				ema = tf.train.ExponentialMovingAverage(decay=0.9)
				ema_apply_op = tf.cond(is_training,
					lambda: ema.apply([batch_mean, batch_var]),
					lambda: tf.no_op())

				def mean_var_with_update():
					with tf.control_dependencies([ema_apply_op]):
						return tf.identity(batch_mean), tf.identity(batch_var)

				mean, var = tf.cond(is_training,
					mean_var_with_update,
					lambda: (ema.average(batch_mean), ema.average(batch_var)))
				normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
			return normed

		self.fc[0] = self.concat
		for i in range(NUM_FC_LAYERS):
			self.fc_weights[i] = tf.get_variable('fc_weights'+str(i), [1,self.fc[i].get_shape().as_list()[2], NUM_FC_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.fc_bias[i] = tf.get_variable('fc_bias'+str(i), [NUM_FC_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.fc[i+1] = tf.nn.conv1d(self.fc[i], self.fc_weights[i], 1, padding='VALID')
			self.fc[i+1] = tf.nn.bias_add(self.fc[i+1], self.fc_bias[i])
			self.fc[i+1] = batch_norm_template(self.fc[i+1],self.is_training_pl,[0,])
			self.fc[i+1] = tf.nn.relu(self.fc[i+1])

		#output
		self.fc_weights[-1] = tf.get_variable('fc_weights'+str(NUM_FC_LAYERS), [1,self.fc[-1].get_shape().as_list()[2], num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.fc_bias[-1] = tf.get_variable('fc_bias'+str(NUM_FC_LAYERS), [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.output = tf.nn.conv1d(self.fc[-1], self.fc_weights[-1], 1, padding='VALID')
		self.output = tf.nn.bias_add(self.output, self.fc_bias[-1])

		#loss functions
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels_pl))
		self.loss = self.class_loss
		self.correct = tf.equal(tf.argmax(self.output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(self.correct, tf.float32)) 

		#optimizer
		self.batch = tf.Variable(0)
		self.learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,self.batch,500,0.5,staircase=True)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss, global_step=self.batch)

def sample_and_group(npoint, radius, nsample, xyz, points):
	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
	else:
		new_points = grouped_xyz
	return new_xyz, new_points, idx, grouped_xyz

#PointNet Set Abstraction (SA) Module
def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope):
	data_format = 'NHWC'
	with tf.variable_scope(scope) as sc:
		new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points)
		kernel = [None]*len(mlp)
		bias = [None]*len(mlp)
		for i, num_out_channel in enumerate(mlp):
			kernel[i] = tf.get_variable('kernel'+str(i), [1,1,new_points.get_shape()[-1].value, num_out_channel], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			bias[i] = tf.get_variable('bias'+str(i), [num_out_channel], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			new_points = tf.nn.conv2d(new_points, kernel[i], [1, 1, 1, 1], padding='VALID')
			new_points = tf.nn.bias_add(new_points, bias[i])
			new_points = tf.nn.relu(new_points)
		new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
		new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
		return new_xyz, new_points, idx

#PointNet Feature Propogation (FP) Module
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope):
	with tf.variable_scope(scope) as sc:
		dist, idx = three_nn(xyz1, xyz2)
		dist = tf.maximum(dist, 1e-10)
		norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
		norm = tf.tile(norm,[1,1,3])
		weight = (1.0/dist) / norm
		interpolated_points = three_interpolate(points2, idx, weight)

		if points1 is not None:
			new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
		else:
			new_points1 = interpolated_points
		new_points1 = tf.expand_dims(new_points1, 2)

		kernel = [None]*len(mlp)
		bias = [None]*len(mlp)
		for i, num_out_channel in enumerate(mlp):
			kernel[i] = tf.get_variable('kernel'+str(i), [1,1,new_points1.get_shape()[-1].value, num_out_channel], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			bias[i] = tf.get_variable('bias'+str(i), [num_out_channel], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			new_points1 = tf.nn.conv2d(new_points1, kernel[i], [1, 1, 1, 1], padding='VALID')
			new_points1 = tf.nn.bias_add(new_points1, bias[i])
			new_points1 = tf.nn.relu(new_points1)
		new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]s
	return new_points1


class PointNet2():
	def __init__(self,batch_size, feature_size, num_class):
		#input_channels = 6
		#self.pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size, input_channels))
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
		#self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size, feature_size))
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())
		l0_xyz = tf.reshape(self.input_pl[:,:3], [1,batch_size,3])
		l0_points = None

		# Layer 1
		l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer1')
		l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer2')
		l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer3')
		l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer4')

		# Feature Propagation layers
		l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], self.is_training_pl, 0, scope='fa_layer1')
		l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], self.is_training_pl, 0, scope='fa_layer2')
		l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], self.is_training_pl, 0, scope='fa_layer3')
		l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], self.is_training_pl, 0, scope='fa_layer4')

		# FC layers
		l0_points = tf.reshape(l0_points, [batch_size, 128])
		kernel1 = tf.get_variable('kernel1', [128, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('bias1', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc1 = tf.matmul(l0_points, kernel1)
		fc1 = tf.nn.bias_add(fc1, bias1)
		fc1 = tf.nn.relu(fc1)
		kernel2 = tf.get_variable('kernel2', [128, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('bias2', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc2 = tf.matmul(fc1, kernel2)
		self.class_output = tf.nn.bias_add(fc2, bias2)

		#LOSS FUNCTIONS
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.labels_pl))
		correct = tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 
		self.loss = self.class_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

def loadFromH5(filename, load_labels=True):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	if load_labels:
		for i in range(len(tmp_points)):
			room.append(tmp_points[i][:,:-2])
			labels.append(tmp_points[i][:,-2].astype(int))
			class_labels.append(tmp_points[i][:,-1].astype(int))
		return room, labels, class_labels
	else:
		return tmp_points

def jitter_data(points, labels):
	output_points = points.copy()
	output_labels = labels.copy()
	for i in range(len(points)):
		if numpy.random.randint(2):
			output_points[i,:,0] = -output_points[i,:,0]
		if numpy.random.randint(2):
			output_points[i,:,1] = -output_points[i,:,1]
		C = numpy.random.rand() * 0.5 + 0.75
		T = numpy.random.rand(3) * 0.4 - 0.2
		output_points[i,:,:3] = output_points[i,:,:3] * C + T
	return output_points, output_labels

def get_acc(emb,lb):
	correct = 0
	for i in range(len(lb)):
		dist = numpy.sum((emb[i] - emb)**2, axis=1)
		order = numpy.argsort(dist)
		correct += lb[i] == lb[order[1]]
	return 1.0 * correct / len(lb)

def get_anova(emb, lb):
	lid = list(set(lb))
	nf = emb.shape[1]
	class_mean = numpy.zeros((len(lid), nf))
	for i in range(len(lid)):
		class_mean[i] = emb[lb==lid[i]].mean(axis=0)
	overall_mean = emb.mean(axis=0)
	between_group = 0
	for i in range(len(lid)):
		num_in_group = numpy.sum(lb==lid[i])
		between_group += numpy.sum((class_mean[i] - overall_mean)**2) * num_in_group
	between_group /= (len(lid) - 1)
	within_group = 0
	for i in range(len(lid)):
		within_group += numpy.sum((emb[lb==lid[i]] - class_mean[i])**2)
	within_group /= (len(lb) - len(lid))
	F = 0 if within_group==0 else between_group / within_group
	return between_group, within_group, F

def get_even_sampling(labels, batch_size, samples_per_instance):
	pool = {}
	for i in set(labels):
		pool[i] = set(numpy.nonzero(labels==i)[0])
	idx = []
	while len(pool) > 0 and len(idx) < batch_size:
		k = pool.keys()
		c = k[numpy.random.randint(len(k))]	
		if len(pool[c]) > samples_per_instance:
			inliers = set(numpy.random.choice(list(pool[c]), samples_per_instance, replace=False))
			idx.extend(inliers)
			pool[c] -= inliers
		else:
			idx.extend(pool[c])
			del pool[c]
	return idx[:batch_size]


if __name__=='__main__':

	VAL_AREA = 1
	mode = 'pointnet'
	for i in range(len(sys.argv)):
		if sys.argv[i] == '--mode':
			mode = sys.argv[i+1]
		if sys.argv[i]=='--area':
			VAL_AREA = int(sys.argv[i+1])
	if mode == 'pointnet2':
		MODEL_PATH = 'models/pointnet2_model'+str(VAL_AREA)+'.ckpt'
	else:
		MODEL_PATH = 'models/pointnet_model'+str(VAL_AREA)+'.ckpt'

	#arrange points into batches of 2048x6
	train_points = []
	train_labels = []
	val_points = []
	val_labels = []
	for AREA in [1,2,3,4,5,6]:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
		for room_id in range(len(all_points)):
			points = all_points[room_id]
			cls_id = all_cls_id[room_id]

			grid_resolution = 1.0
			grid = numpy.round(points[:,:2]/grid_resolution).astype(int)
			grid_set = set([tuple(g) for g in grid])
			for g in grid_set:
				grid_mask = numpy.all(grid==g, axis=1)
				grid_points = points[grid_mask, :6]
				centroid_xy = numpy.array(g)*grid_resolution
				centroid_z = grid_points[:,2].min()
				grid_points[:,:2] -= centroid_xy
				grid_points[:,2] -= centroid_z
				grid_labels = cls_id[grid_mask]

				subset = numpy.random.choice(len(grid_points), NUM_POINT*2, replace=len(grid_points)<NUM_POINT*2)
				if AREA==VAL_AREA:
					val_points.append(grid_points[subset])
					val_labels.append(grid_labels[subset])
				else:
					train_points.append(grid_points[subset])
					train_labels.append(grid_labels[subset])

	train_points = numpy.array(train_points)
	train_labels = numpy.array(train_labels)
	val_points = numpy.array(val_points)
	val_labels = numpy.array(val_labels)
	print('Train Points',train_points.shape)
	print('Train Labels',train_labels.shape)
	print('Validation Points',val_points.shape)
	print('Validation Labels',val_labels.shape)

	with tf.Graph().as_default():
		with tf.device('/gpu:'+str(GPU_INDEX)):
			if mode == 'pointnet2':
				net = PointNet2(BATCH_SIZE,NUM_POINT,NUM_CLASSES)
			else:
				net = PointNet(BATCH_SIZE,NUM_POINT,NUM_CLASSES) 
			saver = tf.train.Saver()

			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True
			config.log_device_placement = False
			sess = tf.Session(config=config)
			init = tf.global_variables_initializer()
			sess.run(init, {net.is_training_pl: True})


			for epoch in range(MAX_EPOCH):
				#shuffle data
				idx = numpy.arange(len(train_labels))
				numpy.random.shuffle(idx)
				shuffled_points = train_points[idx, :, :]
				shuffled_labels = train_labels[idx, :]
				input_points = numpy.zeros((BATCH_SIZE, NUM_POINT, 6))
				input_labels = numpy.zeros((BATCH_SIZE, NUM_POINT))

				#split into batches
				num_batches = int(len(train_labels) / BATCH_SIZE)
				class_loss = []
				class_acc = []
				inner_loss = []
				for batch_id in range(num_batches):
					start_idx = batch_id * BATCH_SIZE
					end_idx = (batch_id + 1) * BATCH_SIZE
					if NUM_POINT == shuffled_points.shape[1]:
						input_points[:] = shuffled_points[start_idx:end_idx,:,:]
						input_labels = shuffled_labels[start_idx:end_idx,:]
					else:
						for i in range(BATCH_SIZE):
							subset = numpy.random.choice(shuffled_points.shape[1], NUM_POINT, replace=False)
							input_points[i,:,:] = shuffled_points[start_idx+i, subset, :]
							input_labels[i,:] = shuffled_labels[start_idx+i,subset]
					input_points, input_labels = jitter_data(input_points, input_labels)
					feed_dict = {net.pointclouds_pl: input_points,
						net.labels_pl: input_labels,
						net.is_training_pl: True}
					
					a1,l1,_ = sess.run([net.class_acc,net.class_loss,net.train_op], feed_dict=feed_dict)
					class_acc.append(a1)
					class_loss.append(l1)
				print('Epoch: %d Loss: %.3f (cls %.3f)'%(epoch,numpy.mean(class_loss), numpy.mean(class_acc)))

				if epoch % VAL_STEP == VAL_STEP - 1:
					#get validation loss
					num_batches = int(len(val_labels) / BATCH_SIZE)
					class_loss = []
					class_acc = []
					inner_loss = []
					for batch_id in range(num_batches):
						start_idx = batch_id * BATCH_SIZE
						end_idx = (batch_id + 1) * BATCH_SIZE
						if NUM_POINT == val_points.shape[1]:
							input_points[:] = val_points[start_idx:end_idx,:,:]
							input_labels = val_labels[start_idx:end_idx,:]
						else:
							for i in range(BATCH_SIZE):
								subset = numpy.random.choice(val_points.shape[1], NUM_POINT, replace=False)
								input_points[i,:,:] = val_points[start_idx+i, subset, :]
								input_labels[i,:] = val_labels[start_idx+i,subset]
						feed_dict = {net.pointclouds_pl: input_points,
							net.labels_pl: input_labels,
							net.is_training_pl: False}
						a1,l1 = sess.run([net.class_acc,net.class_loss], feed_dict=feed_dict)
						class_acc.append(a1)
						class_loss.append(l1)
					print('Validation: %d Loss: %.3f (cls %.3f)'%(epoch,numpy.mean(class_loss), numpy.mean(class_acc)))
			#save trained model
			saver.save(sess, MODEL_PATH)
	
