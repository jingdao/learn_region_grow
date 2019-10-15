import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from metric_loss_ops import triplet_semihard_loss

action_map = numpy.array([
	[0,0,0,0,0,0],
	[-1,0,0,0,0,0],
	[0,0,0,1,0,0],
	[0,-1,0,0,0,0],
	[0,0,0,0,1,0],
	[0,0,-1,0,0,0],
	[0,0,0,0,0,1],
])
action_str = ['no-op','-x','+x','-y','+y','-z','+z']

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

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
	f.close()
	print 'Saved %d points to %s' % (l,filename)

def savePLY(filename, points):
	f = open(filename,'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar r
property uchar g
property uchar b
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

def normalize(stacked_points, stacked_neighbor_points):
	for i in range(len(stacked_points)):
		center = numpy.mean(stacked_points[i][:,:2], axis=0)
		stacked_points[i][:,:2] -= center
		rgb_center = numpy.mean(stacked_points[i][:,3:6], axis=0)
		normal_center = numpy.mean(stacked_points[i][:,6:9], axis=0)
		if len(stacked_neighbor_points[i]) > 0:
			stacked_neighbor_points[i][:,:2] -= center
			stacked_neighbor_points[i][:,3:6] -= rgb_center
			stacked_neighbor_points[i][:,6:9] -= normal_center

class LrgNet:
	def __init__(self,batch_size, num_points, num_neighbor_points, feature_size):
		CONV_CHANNELS = [64,64,64,128,512]
		CONV2_CHANNELS = [256, 128]
		FC_CHANNELS = [256, 128]
		self.kernel = [None]*len(CONV_CHANNELS)
		self.bias = [None]*len(CONV_CHANNELS)
		self.conv = [None]*len(CONV_CHANNELS)
		self.fc = [None]*(len(FC_CHANNELS) + 1)
		self.fc_kernel = [None]*(len(FC_CHANNELS) + 1)
		self.fc_bias = [None]*(len(FC_CHANNELS) + 1)
		self.neighbor_kernel = [None]*(len(CONV_CHANNELS) + len(CONV2_CHANNELS) + 1)
		self.neighbor_bias = [None]*(len(CONV_CHANNELS) + len(CONV2_CHANNELS) + 1)
		self.neighbor_conv = [None]*(len(CONV_CHANNELS) + len(CONV2_CHANNELS) + 1)
		self.tile = [None]*2
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, feature_size))
		self.neighbor_pl = tf.placeholder(tf.float32, shape=(batch_size, num_neighbor_points, feature_size))
		self.class_pl = tf.placeholder(tf.int32, shape=(batch_size, num_neighbor_points))
		self.completeness_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())

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

		#CONVOLUTION LAYERS
		for i in range(len(CONV_CHANNELS)):
			self.kernel[i] = tf.get_variable('lrg_kernel'+str(i), [1, feature_size if i==0 else CONV_CHANNELS[i-1], CONV_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('lrg_bias'+str(i), [CONV_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.nn.conv1d(self.input_pl if i==0 else self.conv[i-1], self.kernel[i], 1, padding='VALID')
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = batch_norm_template(self.conv[i], self.is_training_pl, [0,])
			self.conv[i] = tf.nn.relu(self.conv[i])

		#CONVOLUTION LAYERS FOR NEIGHBOR INPUT
		for i in range(len(CONV_CHANNELS)):
			self.neighbor_kernel[i] = tf.get_variable('lrg_neighbor_kernel'+str(i), [1, feature_size if i==0 else CONV_CHANNELS[i-1], CONV_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.neighbor_bias[i] = tf.get_variable('lrg_neighbor_bias'+str(i), [CONV_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.neighbor_conv[i] = tf.nn.conv1d(self.neighbor_pl if i==0 else self.neighbor_conv[i-1], self.neighbor_kernel[i], 1, padding='VALID')
			self.neighbor_conv[i] = tf.nn.bias_add(self.neighbor_conv[i], self.neighbor_bias[i])
			self.neighbor_conv[i] = batch_norm_template(self.neighbor_conv[i], self.is_training_pl, [0,])
			self.neighbor_conv[i] = tf.nn.relu(self.neighbor_conv[i])

		#MAX POOLING
		self.pool = tf.reduce_max(self.conv[4], axis=1)
		self.neighbor_pool = tf.reduce_max(self.neighbor_conv[4], axis=1)
		self.combined_pool = tf.concat(axis=1, values=[self.pool, self.neighbor_pool])

		##COMPLETENESS BRANCH##
		for i in range(len(FC_CHANNELS)):
			self.fc_kernel[i] = tf.get_variable('lrg_fc_kernel'+str(i), [CONV_CHANNELS[-1]*2 if i==0 else FC_CHANNELS[i-1], FC_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.fc_bias[i] = tf.get_variable('lrg_fc_bias'+str(i), [FC_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.fc[i] = tf.matmul(self.combined_pool if i==0 else self.fc[i-1], self.fc_kernel[i])
			self.fc[i] = tf.nn.bias_add(self.fc[i], self.fc_bias[i])
			self.fc[i] = batch_norm_template(self.fc[i],self.is_training_pl,[0,])
			self.fc[i] = tf.nn.relu(self.fc[i])
		i += 1
		self.fc_kernel[i] = tf.get_variable('lrg_fc_kernel'+str(i), [FC_CHANNELS[-1], 2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.fc_bias[i] = tf.get_variable('lrg_fc_bias'+str(i), [2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[i] = tf.matmul(self.fc[i-1], self.fc_kernel[i])
		self.fc[i] = tf.nn.bias_add(self.fc[i], self.fc_bias[i])
		self.completeness_output = self.fc[i]

		##CLASSIFICATION BRANCH##

		#CONCAT AFTER POOLING
		self.tile[0] = tf.tile(tf.reshape(self.combined_pool,[batch_size,-1,CONV_CHANNELS[-1]*2]) , [1,1,num_neighbor_points])
		self.tile[0] = tf.reshape(self.tile[0],[batch_size,num_neighbor_points,-1])
		self.tile[1] = self.neighbor_conv[1]
		self.concat = tf.concat(axis=2, values=self.tile)

		#CONVOLUTION LAYERS AFTER POOLING
		for i in range(len(CONV2_CHANNELS)):
			kernel_id = i + len(CONV_CHANNELS)
			self.neighbor_kernel[kernel_id] = tf.get_variable('lrg_neighbor_kernel'+str(kernel_id), [1, CONV_CHANNELS[-1]*2 + CONV_CHANNELS[1] if i==0 else CONV2_CHANNELS[i-1], CONV2_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.neighbor_bias[kernel_id] = tf.get_variable('lrg_neighbor_bias'+str(kernel_id), [CONV2_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.neighbor_conv[kernel_id] = tf.nn.conv1d(self.concat if i==0 else self.neighbor_conv[kernel_id-1], self.neighbor_kernel[kernel_id], 1, padding='VALID')
			self.neighbor_conv[kernel_id] = tf.nn.bias_add(self.neighbor_conv[kernel_id], self.neighbor_bias[kernel_id])
			self.neighbor_conv[kernel_id] = batch_norm_template(self.neighbor_conv[kernel_id], self.is_training_pl, [0,])
			self.neighbor_conv[kernel_id] = tf.nn.relu(self.neighbor_conv[kernel_id])
		kernel_id = i + len(CONV_CHANNELS) + 1
		self.neighbor_kernel[kernel_id] = tf.get_variable('lrg_neighbor_kernel'+str(kernel_id), [1, CONV2_CHANNELS[-1], 2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.neighbor_bias[kernel_id] = tf.get_variable('lrg_neighbor_bias'+str(kernel_id), [2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.neighbor_conv[kernel_id] = tf.nn.conv1d(self.neighbor_conv[kernel_id-1], self.neighbor_kernel[kernel_id], 1, padding='VALID')
		self.neighbor_conv[kernel_id] = tf.nn.bias_add(self.neighbor_conv[kernel_id], self.neighbor_bias[kernel_id])
		self.class_output = self.neighbor_conv[kernel_id]

		#LOSS FUNCTIONS
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.class_pl))
		self.class_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.class_pl)), tf.float32))
		pos_mask = tf.where(tf.cast(self.completeness_pl, tf.bool))
		neg_mask = tf.where(tf.cast(1 - self.completeness_pl, tf.bool))
		self.pos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(self.completeness_output, pos_mask), labels=tf.gather_nd(self.completeness_pl, pos_mask)))
		self.neg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(self.completeness_output, neg_mask), labels=tf.gather_nd(self.completeness_pl, neg_mask)))
		self.pos_loss = tf.cond(tf.is_nan(self.pos_loss), lambda: 0.0, lambda: self.pos_loss)
		self.neg_loss = tf.cond(tf.is_nan(self.neg_loss), lambda: 0.0, lambda: self.neg_loss)
		self.completeness_loss = self.pos_loss + self.neg_loss
		correct = tf.equal(tf.argmax(self.completeness_output, -1), tf.to_int64(self.completeness_pl))
		self.completeness_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
		self.loss = self.class_loss + self.completeness_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(1e-4)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

class MCPNet:
	def __init__(self,batch_size, neighbor_size, feature_size, hidden_size, embedding_size):
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, neighbor_size, feature_size))
		self.label_pl = tf.placeholder(tf.int32, shape=(batch_size))

		#NETWORK_WEIGHTS
		kernel1 = tf.get_variable('mcp_kernel1', [1,feature_size,hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('mcp_bias1', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel2 = tf.get_variable('mcp_kernel2', [1,hidden_size,hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('mcp_bias2', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel3 = tf.get_variable('mcp_kernel3', [hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias3 = tf.get_variable('mcp_bias3', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel4 = tf.get_variable('mcp_kernel4', [hidden_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias4 = tf.get_variable('mcp_bias4', [embedding_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

		#MULTI-VIEW CONTEXT POOLING
		neighbor_fc = tf.nn.conv1d(self.input_pl, kernel1, 1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias1)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.nn.conv1d(neighbor_fc, kernel2, 1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias2)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.reduce_max(neighbor_fc, axis=1)

		#FEATURE EMBEDDING BRANCH (for instance label prediction)
		fc3 = tf.matmul(neighbor_fc, kernel3)
		fc3 = tf.nn.bias_add(fc3, bias3)
		fc3 = tf.nn.relu(fc3)
		self.fc4 = tf.matmul(fc3, kernel4)
		self.fc4 = tf.nn.bias_add(self.fc4, bias4)
		self.embeddings = tf.nn.l2_normalize(self.fc4, dim=1)
		self.triplet_loss = triplet_semihard_loss(self.label_pl, self.embeddings)

		#LOSS FUNCTIONS
		self.loss = self.triplet_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)
