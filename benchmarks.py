import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from class_util import classes, class_to_id, class_to_color_rgb
import itertools
import random
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import math
import networkx as nx
from scipy.cluster.vq import vq, kmeans
import time
import matplotlib.pyplot as plt
import scipy.special
from train_pointnet import PointNet

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

numpy.random.seed(0)
TEST_AREAS = [1,2,3,4,5,6,'scannet']
resolution = 0.1
feature_size = 9
NUM_POINT = 1024
mode = 'normal'
if mode=='normal':
	threshold = 0.99
elif mode == 'curvature':
	threshold = 0.99
elif mode=='color':
	threshold = 0.005
elif mode=='embedding':
	threshold = 0.9
else:
	threshold = 0.98
save_results = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []

for i in range(len(sys.argv)):
	if sys.argv[i]=='--mode':
		mode = sys.argv[i+1]
	elif sys.argv[i]=='--area':
		TEST_AREAS = sys.argv[i+1].split(',')
	elif sys.argv[i]=='--threshold':
		threshold = float(sys.argv[i+1])
	elif sys.argv[i]=='--save':
		save_results = True

for AREA in TEST_AREAS:
	tf.reset_default_graph()
	if mode=='pointnet':
		if AREA == 'scannet':
			MODEL_PATH = 'models/pointnet_model3.ckpt'
		else:
			MODEL_PATH = 'models/pointnet_model'+str(AREA)+'.ckpt'
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)
		net = PointNet(1,NUM_POINT,len(classes)) 
		saver = tf.train.Saver()
		saver.restore(sess, MODEL_PATH)
		print('Restored from %s'%MODEL_PATH)
	elif mode=='sgpn':
		pass
	elif mode=='mcpnet':
		pass

	if AREA=='scannet':
		all_points,all_obj_id,all_cls_id = loadFromH5('data/scannet.h5')
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)

	for room_id in range(len(all_points)):
		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]

		#equalize resolution
		equalized_idx = []
		equalized_set = set()
		normal_grid = {}
		for i in range(len(unequalized_points)):
			k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_set:
				equalized_set.add(k)
				equalized_idx.append(i)
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]

		#compute normals
		if mode=='normal' or mode=='curvature':
			normals = []
			curvatures = []
			for i in range(len(points)):
				k = tuple(numpy.round(points[i,:3]/resolution).astype(int))
				neighbors = []
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in normal_grid:
						neighbors.extend(normal_grid[kk])
				accA = numpy.zeros((3,3))
				accB = numpy.zeros(3)
				for n in neighbors:
					p = unequalized_points[n,:3]
					accA += numpy.outer(p,p)
					accB += p
				cov = accA / len(neighbors) - numpy.outer(accB, accB) / len(neighbors)**2
				U,S,V = numpy.linalg.svd(cov)
				# eigenvalues s2<s1<s0
				curvature = S[2] / (S[0] + S[1] + S[2])
				normals.append(numpy.fabs(V[2]))
				curvatures.append(curvature)
			normals = numpy.array(normals)
			curvatures = numpy.array(curvatures)
			points = numpy.hstack((points, normals)).astype(numpy.float32)

		#find connected edges on a voxel grid
		voxel_map = {}
		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		for i in range(len(point_voxels)):
			voxel_map[tuple(point_voxels[i])] = i
		edges = []
		if mode=='normal':
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and normals[voxel_map[kk]].dot(normals[i]) > threshold:
							edges.append([i, voxel_map[kk]])
		elif mode=='curvature':
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and (curvatures[voxel_map[kk]] - curvatures[i]) < threshold:
							edges.append([i, voxel_map[kk]])
		elif mode=='color':
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and numpy.sum((points[voxel_map[kk],3:6] - points[i,3:6])**2) < threshold:
							edges.append([i, voxel_map[kk]])
		elif mode=='pointnet':
			class_labels = numpy.zeros(len(points))
			grid_resolution = 1.0
			grid = numpy.round(points[:,:2]/grid_resolution).astype(int)
			grid_set = set([tuple(g) for g in grid])
			for g in grid_set:
				grid_mask = numpy.all(grid==g, axis=1)
				grid_points = points[grid_mask, :]
				centroid_xy = numpy.array(g)*grid_resolution
				centroid_z = grid_points[:,2].min()
				grid_points[:,:2] -= centroid_xy
				grid_points[:,2] -= centroid_z
				input_points = numpy.zeros((1, NUM_POINT, 6))
				input_points[0,:len(grid_points),:] = grid_points[:NUM_POINT,:6]
				input_points[0,len(grid_points):,:] = grid_points[0,:6]
				cls, = sess.run([net.output], feed_dict={net.pointclouds_pl: input_points, net.is_training_pl: False})
				cls = cls[0].argmax(axis=1)
				class_labels[grid_mask] = cls[:len(grid_points)]

			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and class_labels[voxel_map[kk]]==class_labels[i]:
							edges.append([i, voxel_map[kk]])

		elif mode=='sgpn':
			pass
		elif mode=='mcpnet':
			pass

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
			if mode=='normal':
				print('points shape: ', points.shape)
				print('normals shape: ', normals.shape)
				points[:,3:6] = normals*255
				savePLY('data/normal/%d.ply'%save_id, points)
			elif mode == 'curvature':
				print("shape, ", points.shape)
				print("shape_curvature: ", curvatures.shape)
				points[:,3] = curvatures*255
				points[:,4] = (1-curvatures)*255
				points[:,5] = (curvatures**0.9)*255
				savePLY('data/curvature/%d.ply'%save_id, points)
			elif mode=='pointnet':
				points[:,3:6] = [class_to_color_rgb[c] for c in class_labels]
				savePLY('data/class/%d.ply'%save_id, points)
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			points[:,3:6] = obj_color[cluster_label2,:]
			savePLY('data/results/%d.ply'%save_id, points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))

