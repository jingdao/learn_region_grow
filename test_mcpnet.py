import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import sys
import itertools
import random
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from learn_region_grow_util import *

numpy.random.seed(0)
local_range = 2
resolution = 0.1
batch_size = 1
num_neighbors = 50
neighbor_radii = 0.3
hidden_size = 200
embedding_size = 10
dp_threshold = 0.9
#feature_size = 6
feature_size = 3
TEST_AREAS = [1,2,3,4,5,6,'scannet']
save_results = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []

for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		TEST_AREAS = sys.argv[i+1].split(',')
	elif sys.argv[i]=='--save':
		save_results = True

for AREA in TEST_AREAS:
	tf.reset_default_graph()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)
	net = MCPNet(batch_size, num_neighbors, feature_size, hidden_size, embedding_size)
	saver = tf.train.Saver()
	if AREA=='scannet':
		MODEL_PATH = 'models/mcpnet_model1.ckpt'
	else:
		MODEL_PATH = 'models/mcpnet_model%s.ckpt'%AREA
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_PATH)
	print('Restored from %s'%MODEL_PATH)

	if AREA=='scannet':
		all_points,all_obj_id,all_cls_id = loadFromH5('data/scannet.h5')
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)
	stacked_points = []
	stacked_neighbor_points = []
	stacked_labels = []

	for room_id in range(len(all_points)):
#	for room_id in [0]:
		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]
		centroid = 0.5 * (unequalized_points[:,:2].min(axis=0) + unequalized_points[:,:2].max(axis=0))
		unequalized_points[:,:2] -= centroid
		unequalized_points[:,2] -= unequalized_points[:,2].min()

		#equalize resolution
		equalized_idx = []
		equalized_map = {}
		coarse_map = {}
		unequalized_idx = []
		for i in range(len(unequalized_points)):
			k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_map:
				equalized_map[k] = len(equalized_idx)
				equalized_idx.append(i)
				kk = tuple(numpy.round(unequalized_points[i,:3]/neighbor_radii).astype(int))
				if not kk in coarse_map:
					coarse_map[kk] = []
				coarse_map[kk].append(equalized_map[k])
			unequalized_idx.append(equalized_map[k])
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]

		#compute neighbors for each point
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

		#compute embedding for each point
		embeddings = numpy.zeros((len(points), embedding_size), dtype=float)
		input_points = numpy.zeros((batch_size, num_neighbors, feature_size), dtype=float)
		num_batches = 0
		for i in range(len(points)):
			input_points[0,:,:] = neighbor_array[i, :, :feature_size]
			emb_val = sess.run(net.embeddings, {net.input_pl:input_points})
			embeddings[i] = emb_val
			num_batches += 1

		#find connected edges on a voxel grid
		voxel_map = {}
		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		for i in range(len(point_voxels)):
			voxel_map[tuple(point_voxels[i])] = i
		edges = []
		for i in range(len(point_voxels)):
			k = tuple(point_voxels[i])
			for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
				if offset!=(0,0,0):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in voxel_map and embeddings[voxel_map[kk]].dot(embeddings[i]) > dp_threshold:
						edges.append([i, voxel_map[kk]])

		#calculate connected components from edges
		G = nx.Graph(edges)
		clusters = nx.connected_components(G)
		clusters = [list(c) for c in clusters]
		cluster_label = numpy.zeros(len(point_voxels),dtype=int)
		min_cluster_size = 10
		cluster_id = 1
		for i in range(len(clusters)):
			if len(clusters[i]) > min_cluster_size:
				cluster_label[clusters[i]] = cluster_id
				cluster_id += 1
	
		#calculate statistics 
		gt_match = 0
		match_id = 0
		dt_match = numpy.zeros(cluster_label.max(), dtype=bool)
		cluster_label2 = numpy.zeros(len(cluster_label), dtype=int)
		room_iou = []
		for i in set(obj_id):
			best_iou = 0
			for j in range(1, cluster_label.max()+1):
				if not dt_match[j-1]:
					iou = 1.0 * numpy.sum(numpy.logical_and(obj_id==i, cluster_label==j)) / numpy.sum(numpy.logical_or(obj_id==i, cluster_label==j))
					best_iou = max(best_iou, iou)
					if iou > 0.5:
						dt_match[j-1] = True
						gt_match += 1
						cluster_label2[cluster_label==j] = i
						break
			room_iou.append(best_iou)
		for j in range(1,cluster_label.max()+1):
			if not dt_match[j-1]:
				cluster_label2[cluster_label==j] = j + obj_id.max()
		prc = numpy.mean(dt_match)
		rcl = 1.0 * gt_match / len(set(obj_id))
		room_iou = numpy.mean(room_iou)

		nmi = normalized_mutual_info_score(obj_id,cluster_label)
		ami = adjusted_mutual_info_score(obj_id,cluster_label)
		ars = adjusted_rand_score(obj_id,cluster_label)
		agg_nmi.append(nmi)
		agg_ami.append(ami)
		agg_ars.append(ars)
		agg_prc.append(prc)
		agg_rcl.append(rcl)
		agg_iou.append(room_iou)
		print("Area %s room %d NMI: %.2f AMI: %.2f ARS: %.2f PRC: %.2f RCL: %.2f IOU: %.2f"%(str(AREA), room_id, nmi,ami,ars, prc, rcl, room_iou))

		#save point cloud results to file
		if save_results:
			X_embedded = PCA(n_components=3).fit_transform(embeddings)
			obj_color = (X_embedded - X_embedded.min(axis=0)) / (X_embedded.max(axis=0) - X_embedded.min(axis=0)) * 255
			unequalized_points[:,3:6] = obj_color[unequalized_idx]
			savePLY('data/embedding/%d.ply'%save_id, unequalized_points)
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
			savePLY('data/results/%d.ply'%save_id, unequalized_points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))
