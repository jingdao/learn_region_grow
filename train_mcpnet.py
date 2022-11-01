import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import sys
import itertools
import random
from learn_region_grow_util import *

numpy.random.seed(0)
local_range = 2
resolution = 0.1
num_neighbors = 50
neighbor_radii = 0.3
batch_size = 256
hidden_size = 200
embedding_size = 10
#feature_size = 6
feature_size = 3
max_epoch = 50
samples_per_instance = 16
stage_data = False

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

if stage_data:
	for AREA in range(1,7):
#	for AREA in [3]:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
		stacked_points = []
		stacked_neighbor_points = []
		stacked_labels = []

		for room_id in range(len(all_points)):
#		for room_id in [0]:
			unequalized_points = all_points[room_id]
			obj_id = all_obj_id[room_id]
			cls_id = all_cls_id[room_id]
			centroid = 0.5 * (unequalized_points[:,:2].min(axis=0) + unequalized_points[:,:2].max(axis=0))
			unequalized_points[:,:2] -= centroid
			unequalized_points[:,2] -= unequalized_points[:,2].min()

			#equalize resolution
			equalized_idx = []
			equalized_set = set()
			coarse_map = {}
			for i in range(len(unequalized_points)):
				k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
				if not k in equalized_set:
					equalized_set.add(k)
					equalized_idx.append(i)
					kk = tuple(numpy.round(unequalized_points[i,:3]/neighbor_radii).astype(int))
					if not kk in coarse_map:
						coarse_map[kk] = []
					coarse_map[kk].append(len(equalized_set)-1)
			points = unequalized_points[equalized_idx]
			obj_id = obj_id[equalized_idx]
			cls_id = cls_id[equalized_idx]

			neighbor_array = numpy.zeros((len(points), num_neighbors, 6), dtype=float)
			for i in range(len(points)):
				p = points[i,:6]
				k = tuple(numpy.round(points[i,:3]/neighbor_radii).astype(int))
				neighbors = []
				for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in coarse_map:
						neighbors.extend(coarse_map[kk])
				neighbors = numpy.random.choice(neighbors, num_neighbors, replace=len(neighbors)<num_neighbors)
				neighbors = points[neighbors, :6].copy()
				neighbors -= p
				neighbor_array[i,:,:] = neighbors

			available = numpy.ones(len(points), dtype=bool)
			num_batches = 0
			for i in range(len(points)):
				if not available[i]:
					continue
				center_position = points[i,:2]
				tmp_range = local_range
				local_mask = []
				while True:
					local_mask = numpy.sum((points[:,:2]-center_position)**2, axis=1) < tmp_range * tmp_range
					local_mask = numpy.logical_and(local_mask, available)
					local_mask = numpy.nonzero(local_mask)[0]
					if len(local_mask) >= batch_size*2:
						break
					else:
						tmp_range *= 1.5
				local_mask = numpy.random.choice(local_mask, batch_size*2, replace=False)
				stacked_points.append(points[local_mask, 2:6])
				stacked_neighbor_points.append(neighbor_array[local_mask, :, :])
				stacked_labels.append(obj_id[local_mask])
				available[local_mask] = False
				num_batches += 1
				if numpy.sum(available) < batch_size*2:
					break

			print('AREA %d room %d %d points %d batches'%(AREA, room_id, len(points), num_batches))
			
		h5_fout = h5py.File('data/mcp_area%s.h5'%AREA,'w')
		h5_fout.create_dataset( 'points', data=numpy.array(stacked_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
		h5_fout.create_dataset( 'neighbor_points', data=numpy.array(stacked_neighbor_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
		h5_fout.create_dataset( 'labels', data=stacked_labels, compression='gzip', compression_opts=4, dtype=numpy.int32)
		h5_fout.close()
	sys.exit(1)

VAL_AREA = 1
for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.compat.v1.Session(config=config)
net = MCPNet(batch_size, num_neighbors, feature_size, hidden_size, embedding_size)
saver = tf.compat.v1.train.Saver()
MODEL_PATH = 'models/mcpnet_model%d.ckpt'%(VAL_AREA)

train_points, train_obj_id = None,None
val_points, val_obj_id = None,None

for AREA in range(1,7):
	f = h5py.File('data/mcp_area%d.h5'%AREA,'r')
	print('Loading %s ...'%f.filename)
	if AREA == VAL_AREA:
		val_points = f['neighbor_points'][:,:,:,:feature_size]
		val_obj_id = f['labels'][:]
	else:
		if train_points is None:
			train_points = f['neighbor_points'][:,:,:,:feature_size]
			train_obj_id = f['labels'][:]
		else:
			train_points = numpy.vstack((train_points, f['neighbor_points'][:,:,:,:feature_size]))
			train_obj_id = numpy.vstack((train_obj_id, f['labels'][:]))
	f.close()

print('train',len(train_points),train_points[0].shape)
print('val',len(val_points), val_points[0].shape)
init = tf.compat.v1.global_variables_initializer()
sess.run(init, {})
for epoch in range(max_epoch):
	loss_arr = []
	acc_arr = []
	bg_arr = []
	wg_arr = []
	f_arr = []
	for i in random.sample(xrange(len(train_points)), len(train_points)):
		idx = get_even_sampling(train_obj_id[i], batch_size,samples_per_instance)
		input_points = train_points[i][idx, :, :]
		input_labels = train_obj_id[i][idx]
		_, loss_val, emb_val = sess.run([net.train_op, net.loss, net.embeddings], {net.input_pl:input_points, net.label_pl:input_labels})
		acc = get_acc(emb_val, input_labels)
		bg,wg,f = get_anova(emb_val, input_labels)
		loss_arr.append(loss_val)
		acc_arr.append(acc)
		bg_arr.append(bg)
		wg_arr.append(wg)
		f_arr.append(f)
	print("Epoch %d loss %.2f acc %.2f bg %.2f wg %.2f F %.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(acc_arr),numpy.mean(bg_arr),numpy.mean(wg_arr),numpy.mean(f_arr)))

	if epoch%10==9:
		loss_arr = []
		acc_arr = []
		bg_arr = []
		wg_arr = []
		f_arr = []
		for i in random.sample(xrange(len(val_points)), len(val_points)):
			idx = get_even_sampling(val_obj_id[i], batch_size,samples_per_instance)
			input_points = val_points[i][idx, :, :]
			input_labels = val_obj_id[i][idx]
			loss_val, emb_val = sess.run([net.loss, net.embeddings], {net.input_pl:input_points, net.label_pl:input_labels})
			acc = get_acc(emb_val, input_labels)
			bg,wg,f = get_anova(emb_val, input_labels)
			loss_arr.append(loss_val)
			acc_arr.append(acc)
			bg_arr.append(bg)
			wg_arr.append(wg)
			f_arr.append(f)
		print("Validation %d loss %.2f acc %.2f bg %.2f wg %.2f F %.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(acc_arr),numpy.mean(bg_arr),numpy.mean(wg_arr),numpy.mean(f_arr)))

saver.save(sess, MODEL_PATH)

