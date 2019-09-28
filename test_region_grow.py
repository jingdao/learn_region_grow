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
from learn_region_grow_util import *

numpy.random.seed(0)
NUM_POINT = 512
NUM_NEIGHBOR_POINT = 512
FEATURE_SIZE = 9
TEST_AREAS = [1,2,3,4,5,6,'scannet']
resolution = 0.1
threshold = 0.5
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
	MODEL_PATH = 'models/lrgnet_model%s.ckpt'%AREA
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)
	net = LrgNet(1, NUM_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_PATH)
	print('Restored from %s'%MODEL_PATH)

	if AREA=='scannet':
		all_points,all_obj_id,all_cls_id = loadFromH5('data/scannet.h5')
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)

	for room_id in range(len(all_points)):
#	for room_id in [0]:
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
		normals = []
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
			normals.append(numpy.fabs(V[2]))
		normals = numpy.array(normals)
		points = numpy.hstack((points, normals)).astype(numpy.float32)

		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		cluster_label = numpy.zeros(len(points), dtype=int)
		cluster_id = 1
		visited = numpy.zeros(len(point_voxels), dtype=bool)
		input_points = numpy.zeros((1, NUM_POINT, FEATURE_SIZE), dtype=numpy.float32)
		neighbor_points = numpy.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=numpy.float32)
		input_classes = numpy.zeros((1, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
		#iterate over each object in the room
		for seed_id in range(len(point_voxels)):
			if visited[seed_id]:
				continue
			seed_voxel = point_voxels[seed_id]
			target_id = obj_id[seed_id]
			gt_mask = obj_id==target_id
			obj_voxels = point_voxels[gt_mask]
			obj_voxel_set = set([tuple(p) for p in obj_voxels])
			original_minDims = obj_voxels.min(axis=0)
			original_maxDims = obj_voxels.max(axis=0)
#			print('original',numpy.sum(gt_mask), original_minDims, original_maxDims)
			currentMask = numpy.zeros(len(points), dtype=bool)
			currentMask[seed_id] = True
			minDims = seed_voxel.copy()
			maxDims = seed_voxel.copy()
			steps = 0

			#perform region growing
			while True:

				#determine the current points and the neighboring points
				currentPoints = points[currentMask, :].copy()
				expandPoints = []
				expandClass = []
				for a in range(len(action_map)):
					if a==0:
						mask = numpy.logical_and(numpy.all(point_voxels>=minDims,axis=1), numpy.all(point_voxels<=maxDims, axis=1))
						mask = numpy.logical_and(mask, numpy.logical_not(currentMask))
					else:
						newMinDims = minDims.copy()	
						newMaxDims = maxDims.copy()	
						expand_dim = numpy.nonzero(action_map[a])[0][0] % 3
						if numpy.sum(action_map[a])>0:
							newMinDims[expand_dim] = newMaxDims[expand_dim] = maxDims[expand_dim]+1
						else:
							newMinDims[expand_dim] = newMaxDims[expand_dim] = minDims[expand_dim]-1
						mask = numpy.logical_and(numpy.all(point_voxels>=newMinDims,axis=1), numpy.all(point_voxels<=newMaxDims, axis=1))
					mask = numpy.logical_and(mask, numpy.logical_not(visited))
					expandPoints.extend(points[mask,:].copy())
					#determine which neighboring points should be added
					expandClass.extend(obj_id[mask] == target_id)

				if len(expandPoints)==0: #no neighbors (early termination)
					visited[currentMask] = True
					if numpy.sum(currentMask) > 10:
						cluster_label[currentMask] = cluster_id
						cluster_id += 1
					iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
					print('room %d target %3d: step %3d %4d/%4d points IOU %.2f cls %.3f cmpl %.2f noneighbor'%(room_id, target_id, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, cls_acc, cmpl_conf))
					break 

				subset = numpy.random.choice(len(currentPoints), NUM_POINT, replace=len(currentPoints)<NUM_POINT)
				input_points[0,:,:] = currentPoints[subset, :]
				center = numpy.mean(input_points[0,:,:2], axis=0)
				rgb_center = numpy.mean(input_points[0,:,3:6], axis=0)
				normal_center = numpy.mean(input_points[0,:,6:9], axis=0)
				if len(expandPoints) >= NUM_NEIGHBOR_POINT:
					subset = numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = range(len(expandPoints)) + list(numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
				neighbor_points[0,:,:] = numpy.array(expandPoints)[subset, :]
				input_points[0,:,:2] -= center
				neighbor_points[0,:,:2] -= center
#				scale = numpy.max(numpy.abs(neighbor_points[0,:,:2]))
#				neighbor_points[0,:,:3] /= scale
#				input_points[0,:,:3] /= scale
				neighbor_points[0,:,3:6] -= rgb_center
				neighbor_points[0,:,6:9] -= normal_center
				input_classes[0,:] = numpy.array(expandClass)[subset]
				input_complete = numpy.zeros(1,dtype=numpy.int32)
				ls, cls, cls_acc, cmpl, cmpl_acc = sess.run([net.loss, net.class_output, net.class_acc, net.completeness_output, net.completeness_acc],
					{net.input_pl:input_points, net.neighbor_pl:neighbor_points, net.completeness_pl:input_complete, net.class_pl:input_classes})

				cls_conf = scipy.special.softmax(cls[0], axis=-1)[:,1]
				cls_mask = cls_conf > threshold
#				cls_mask = input_classes[0].astype(bool)
				cmpl_conf = scipy.special.softmax(cmpl[0], axis=-1)[1]
				validPoints = neighbor_points[0,:,:][cls_mask]
#				validPoints[:,:3] *= scale
				validPoints[:,:2] += center
				validVoxels = numpy.round(validPoints[:,:3]/resolution).astype(int)
				expandSet = set([tuple(p) for p in validVoxels])
				for i in range(len(point_voxels)):
					if tuple(point_voxels[i]) in expandSet:
						currentMask[i] = True
#				print(numpy.sum(currentMask), numpy.sum(gt_mask), len(expandPoints), numpy.sum(expandClass), numpy.sum(input_classes),len(expandSet), cls_acc, cmpl_conf)

#				if numpy.sum(currentMask) == numpy.sum(gt_mask): #completed
				if cmpl_conf > 0.5:
					visited[currentMask] = True
					cluster_label[currentMask] = cluster_id
					cluster_id += 1
					iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
					print('room %d target %3d: step %3d %4d/%4d points IOU %.2f cls %.3f cmpl %.2f'%(room_id, target_id, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, cls_acc, cmpl_conf))
					break 
				else:
					if len(expandSet) > 0: #continue growing
						minDims = point_voxels[currentMask, :].min(axis=0)
						maxDims = point_voxels[currentMask, :].max(axis=0)
					else: #no matching neighbors (early termination)
						visited[currentMask] = True
						if numpy.sum(currentMask) > 10:
							cluster_label[currentMask] = cluster_id
							cluster_id += 1
						iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
						print('room %d target %3d: step %3d %4d/%4d points IOU %.2f cls %.3f cmpl %.2f noexpand'%(room_id, target_id, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, cls_acc, cmpl_conf))
						break 
				steps += 1

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
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			points[:,3:6] = obj_color[cluster_label2,:]
			savePLY('data/results/%d.ply'%save_id, points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))

