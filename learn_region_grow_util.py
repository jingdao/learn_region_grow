import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
from metric_loss_ops import triplet_semihard_loss

tf.compat.v1.disable_eager_execution()

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
	print('Saved %d points to %s' % (l,filename))

def savePLY(filename, points):
	f = open(filename,'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

class LrgNet:
	def __init__(self,batch_size, seq_len, num_inlier_points, num_neighbor_points, feature_size, lite=0):
		if lite==0 or lite is None:
			CONV_CHANNELS = [64,64,64,128,512]
			CONV2_CHANNELS = [256, 128]
		elif lite==1:
			CONV_CHANNELS = [64,64]
			CONV2_CHANNELS = [64]
		elif lite==2:
			CONV_CHANNELS = [64,64,256]
			CONV2_CHANNELS = [64,64]
		self.kernel = [None]*len(CONV_CHANNELS)
		self.bias = [None]*len(CONV_CHANNELS)
		self.conv = [None]*len(CONV_CHANNELS)
		self.neighbor_kernel = [None]*len(CONV_CHANNELS)
		self.neighbor_bias = [None]*len(CONV_CHANNELS)
		self.neighbor_conv = [None]*len(CONV_CHANNELS)
		self.add_kernel = [None]*(len(CONV2_CHANNELS) + 1)
		self.add_bias = [None]*(len(CONV2_CHANNELS) + 1)
		self.add_conv = [None]*(len(CONV2_CHANNELS) + 1)
		self.remove_kernel = [None]*(len(CONV2_CHANNELS) + 1)
		self.remove_bias = [None]*(len(CONV2_CHANNELS) + 1)
		self.remove_conv = [None]*(len(CONV2_CHANNELS) + 1)
		self.inlier_tile = [None]*2
		self.neighbor_tile = [None]*2
		self.inlier_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_len, num_inlier_points, feature_size))
		self.neighbor_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_len, num_neighbor_points, feature_size))
		self.add_mask_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size*seq_len, num_neighbor_points))
		self.remove_mask_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size*seq_len, num_inlier_points))

		#CONVOLUTION LAYERS FOR INLIER SET
		for i in range(len(CONV_CHANNELS)):
			self.kernel[i] = tf.compat.v1.get_variable('lrg_kernel'+str(i), [1, feature_size if i==0 else CONV_CHANNELS[i-1], CONV_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.bias[i] = tf.compat.v1.get_variable('lrg_bias'+str(i), [CONV_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.nn.conv1d(input=self.inlier_pl if i==0 else self.conv[i-1], filters=self.kernel[i], stride=1, padding='VALID')
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = tf.nn.relu(self.conv[i])

		#CONVOLUTION LAYERS FOR NEIGHBOR SET
		for i in range(len(CONV_CHANNELS)):
			self.neighbor_kernel[i] = tf.compat.v1.get_variable('lrg_neighbor_kernel'+str(i), [1, feature_size if i==0 else CONV_CHANNELS[i-1], CONV_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.neighbor_bias[i] = tf.compat.v1.get_variable('lrg_neighbor_bias'+str(i), [CONV_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.neighbor_conv[i] = tf.nn.conv1d(input=self.neighbor_pl if i==0 else self.neighbor_conv[i-1], filters=self.neighbor_kernel[i], stride=1, padding='VALID')
			self.neighbor_conv[i] = tf.nn.bias_add(self.neighbor_conv[i], self.neighbor_bias[i])
			self.neighbor_conv[i] = tf.nn.relu(self.neighbor_conv[i])

		#MAX POOLING
		self.pool = tf.reduce_max(input_tensor=self.conv[-1], axis=1)
		self.neighbor_pool = tf.reduce_max(input_tensor=self.neighbor_conv[-1], axis=1)
		self.combined_pool = tf.concat(axis=1, values=[self.pool, self.neighbor_pool])
		self.pooled_feature = self.combined_pool

		#CONCAT AFTER POOLING
		self.inlier_tile[0] = tf.tile(tf.reshape(self.pooled_feature,[batch_size*seq_len,-1,CONV_CHANNELS[-1]*2]) , [1,1,num_inlier_points])
		self.inlier_tile[0] = tf.reshape(self.inlier_tile[0],[batch_size*seq_len,num_inlier_points,-1])
		self.inlier_tile[1] = self.conv[1]
		self.inlier_concat = tf.concat(axis=2, values=self.inlier_tile)
		self.neighbor_tile[0] = tf.tile(tf.reshape(self.pooled_feature,[batch_size*seq_len,-1,CONV_CHANNELS[-1]*2]) , [1,1,num_neighbor_points])
		self.neighbor_tile[0] = tf.reshape(self.neighbor_tile[0],[batch_size*seq_len,num_neighbor_points,-1])
		self.neighbor_tile[1] = self.neighbor_conv[1]
		self.neighbor_concat = tf.concat(axis=2, values=self.neighbor_tile)

		#CONVOLUTION LAYERS AFTER POOLING
		for i in range(len(CONV2_CHANNELS)):
			self.add_kernel[i] = tf.compat.v1.get_variable('lrg_add_kernel'+str(i), [1, CONV_CHANNELS[-1]*2 + CONV_CHANNELS[1] if i==0 else CONV2_CHANNELS[i-1], CONV2_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.add_bias[i] = tf.compat.v1.get_variable('lrg_add_bias'+str(i), [CONV2_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.add_conv[i] = tf.nn.conv1d(input=self.neighbor_concat if i==0 else self.add_conv[i-1], filters=self.add_kernel[i], stride=1, padding='VALID')
			self.add_conv[i] = tf.nn.bias_add(self.add_conv[i], self.add_bias[i])
			self.add_conv[i] = tf.nn.relu(self.add_conv[i])
		i += 1
		self.add_kernel[i] = tf.compat.v1.get_variable('lrg_add_kernel'+str(i), [1, CONV2_CHANNELS[-1], 2], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		self.add_bias[i] = tf.compat.v1.get_variable('lrg_add_bias'+str(i), [2], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		self.add_conv[i] = tf.nn.conv1d(input=self.add_conv[i-1], filters=self.add_kernel[i], stride=1, padding='VALID')
		self.add_conv[i] = tf.nn.bias_add(self.add_conv[i], self.add_bias[i])
		self.add_output = self.add_conv[i]

		for i in range(len(CONV2_CHANNELS)):
			self.remove_kernel[i] = tf.compat.v1.get_variable('lrg_remove_kernel'+str(i), [1, CONV_CHANNELS[-1]*2 + CONV_CHANNELS[1] if i==0 else CONV2_CHANNELS[i-1], CONV2_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.remove_bias[i] = tf.compat.v1.get_variable('lrg_remove_bias'+str(i), [CONV2_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.remove_conv[i] = tf.nn.conv1d(input=self.inlier_concat if i==0 else self.remove_conv[i-1], filters=self.remove_kernel[i], stride=1, padding='VALID')
			self.remove_conv[i] = tf.nn.bias_add(self.remove_conv[i], self.remove_bias[i])
			self.remove_conv[i] = tf.nn.relu(self.remove_conv[i])
		i += 1
		self.remove_kernel[i] = tf.compat.v1.get_variable('lrg_remove_kernel'+str(i), [1, CONV2_CHANNELS[-1], 2], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		self.remove_bias[i] = tf.compat.v1.get_variable('lrg_remove_bias'+str(i), [2], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		self.remove_conv[i] = tf.nn.conv1d(input=self.remove_conv[i-1], filters=self.remove_kernel[i], stride=1, padding='VALID')
		self.remove_conv[i] = tf.nn.bias_add(self.remove_conv[i], self.remove_bias[i])
		self.remove_output = self.remove_conv[i]

		#LOSS FUNCTIONS
		def weighted_cross_entropy(logit, label):
			pos_mask = tf.compat.v1.where(tf.cast(label, tf.bool))
			neg_mask = tf.compat.v1.where(tf.cast(1 - label, tf.bool))
			pos_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logit, pos_mask), labels=tf.gather_nd(label, pos_mask)))
			neg_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logit, neg_mask), labels=tf.gather_nd(label, neg_mask)))
			pos_loss = tf.cond(pred=tf.math.is_nan(pos_loss), true_fn=lambda: 0.0, false_fn=lambda: pos_loss)
			neg_loss = tf.cond(pred=tf.math.is_nan(neg_loss), true_fn=lambda: 0.0, false_fn=lambda: neg_loss)
			return pos_loss + neg_loss

		self.add_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.add_output, labels=self.add_mask_pl))
		self.add_acc = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=self.add_output, axis=-1), tf.cast(self.add_mask_pl, dtype=tf.int64)), tf.float32))
		TP = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(tf.equal(tf.argmax(input=self.add_output, axis=-1), 1), tf.equal(self.add_mask_pl, 1)), tf.float32))
		self.add_prc = TP / (tf.cast(tf.reduce_sum(input_tensor=tf.argmax(input=self.add_output, axis=-1)), tf.float32) + 1)
		self.add_rcl = TP / (tf.cast(tf.reduce_sum(input_tensor=self.add_mask_pl), tf.float32) + 1)
		self.remove_loss = weighted_cross_entropy(self.remove_output, self.remove_mask_pl)
		self.remove_acc = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=self.remove_output, axis=-1), tf.cast(self.remove_mask_pl, dtype=tf.int64)), tf.float32))
		self.remove_mask = tf.nn.softmax(self.remove_output, axis=-1)[:, :, 1] > 0.5
		TP = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(self.remove_mask, tf.equal(self.remove_mask_pl, 1)), tf.float32))
		self.remove_prc = TP / (tf.reduce_sum(input_tensor=tf.cast(self.remove_mask, tf.float32)) + 1)
		self.remove_rcl = TP / (tf.cast(tf.reduce_sum(input_tensor=self.remove_mask_pl), tf.float32) + 1)

		self.loss = self.add_loss + self.remove_loss
		batch = tf.Variable(0)
		optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

class MCPNet:
	def __init__(self,batch_size, neighbor_size, feature_size, hidden_size, embedding_size):
		self.input_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, feature_size-2))
		self.label_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))
		self.neighbor_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, neighbor_size, feature_size))

		#NETWORK_WEIGHTS
		kernel1 = tf.compat.v1.get_variable('mcp_kernel1', [1,feature_size,hidden_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias1 = tf.compat.v1.get_variable('mcp_bias1', [hidden_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		kernel2 = tf.compat.v1.get_variable('mcp_kernel2', [1,hidden_size,hidden_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias2 = tf.compat.v1.get_variable('mcp_bias2', [hidden_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		kernel3 = tf.compat.v1.get_variable('mcp_kernel3', [feature_size-2+hidden_size, hidden_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias3 = tf.compat.v1.get_variable('mcp_bias3', [hidden_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		kernel4 = tf.compat.v1.get_variable('mcp_kernel4', [hidden_size, embedding_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias4 = tf.compat.v1.get_variable('mcp_bias4', [embedding_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		self.kernels = [kernel1, kernel2, kernel3, kernel4]
		self.biases = [bias1, bias2, bias3, bias4]

		#MULTI-VIEW CONTEXT POOLING
		neighbor_fc = tf.nn.conv1d(input=self.neighbor_pl, filters=kernel1, stride=1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias1)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.nn.conv1d(input=neighbor_fc, filters=kernel2, stride=1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias2)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.reduce_max(input_tensor=neighbor_fc, axis=1)
		concat = tf.concat(axis=1, values=[self.input_pl, neighbor_fc])

		#FEATURE EMBEDDING BRANCH (for instance label prediction)
		fc3 = tf.matmul(concat, kernel3)
		fc3 = tf.nn.bias_add(fc3, bias3)
		fc3 = tf.nn.relu(fc3)
		self.fc4 = tf.matmul(fc3, kernel4)
		self.fc4 = tf.nn.bias_add(self.fc4, bias4)
		self.embeddings = tf.nn.l2_normalize(self.fc4, axis=1)
		self.triplet_loss = triplet_semihard_loss(self.label_pl, self.embeddings)

		#LOSS FUNCTIONS
		self.loss = self.triplet_loss
		batch = tf.Variable(0)
		optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)
