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
import glob
from class_util import classes

numpy.random.seed(0)
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
BEAM_WIDTH = 3
SEARCH_WIDTH = 3
FEATURE_SIZE = 13
TEST_AREAS = ['1','2','3','4','5','6','scannet']
resolution = 0.1
add_threshold = 0.5
rmv_threshold = 0.5
cluster_threshold = 10
save_results = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []
scoring = 'np'

for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		TEST_AREAS = sys.argv[i+1].split(',')
	elif sys.argv[i]=='--save':
		save_results = True
	elif sys.argv[i]=='--scoring':
		scoring = sys.argv[i+1]

for AREA in TEST_AREAS:
	tf.reset_default_graph()
	if AREA=='scannet':
		MODEL_PATH = 'models/lrgnet_model%s.ckpt'%'5'
	else:
		MODEL_PATH = 'models/lrgnet_model%s.ckpt'%AREA
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)
	net = LrgNet(1, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_PATH)
	print('Restored from %s'%MODEL_PATH)
	room_name=[s.split('/')[-1] for s in glob.glob('/home/jd/Documents/GROMI_Deep_Learning/data/Stanford3dDataset_v1.2/Area_%s/*' % AREA)]
	room_name=[s for s in room_name if '.' not in s]

	if AREA=='synthetic':
		all_points,all_obj_id,all_cls_id = loadFromH5('data/synthetic_test.h5')
	elif AREA=='scannet':
		all_points,all_obj_id,all_cls_id = loadFromH5('data/scannet.h5')
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)

#	for room_id in range(len(all_points)):
	for room_id in [0]:
		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]

		#equalize resolution
		equalized_idx = []
		unequalized_idx = []
		equalized_map = {}
		normal_grid = {}
		for i in range(len(unequalized_points)):
			k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_map:
				equalized_map[k] = len(equalized_idx)
				equalized_idx.append(i)
			unequalized_idx.append(equalized_map[k])
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]
		xyz = points[:,:3]
		rgb = points[:,3:6]
		room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

		#compute normals
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
			normals.append(numpy.fabs(V[2]))
			curvature = S[2] / (S[0] + S[1] + S[2])
			curvatures.append(numpy.fabs(curvature))
		curvatures = numpy.array(curvatures)
		curvatures = curvatures/curvatures.max()
		normals = numpy.array(normals)
		points = numpy.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(numpy.float32)

		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		cluster_label = numpy.zeros(len(points), dtype=int)
		cluster_id = 1
		visited = numpy.zeros(len(point_voxels), dtype=bool)
		inlier_points = numpy.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=numpy.float32)
		neighbor_points = numpy.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=numpy.float32)
		input_add = numpy.zeros((1, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
		input_remove = numpy.zeros((1, NUM_INLIER_POINT), dtype=numpy.int32)
		#iterate over each object in the room
		for seed_id in numpy.arange(len(points))[numpy.argsort(curvatures)]:
			if visited[seed_id]:
				continue
			seed_voxel = point_voxels[seed_id]
			target_id = obj_id[seed_id]
			target_class = classes[cls_id[numpy.nonzero(obj_id==target_id)[0][0]]]
			gt_mask = obj_id==target_id
			obj_voxels = point_voxels[gt_mask]
			obj_voxel_set = set([tuple(p) for p in obj_voxels])
			original_minDims = obj_voxels.min(axis=0)
			original_maxDims = obj_voxels.max(axis=0)
			currentMask = numpy.zeros(len(points), dtype=bool)
			currentMask[seed_id] = True
			minDims = seed_voxel.copy()
			maxDims = seed_voxel.copy()
			seqMinDims = minDims
			seqMaxDims = maxDims
			steps = 0
			stuck = 0
			bestScore = 0
			bestMask = currentMask
			Q = [(0, currentMask)]
			qid = 0
			newQ = []

			#perform region growing
			while len(Q) > 0:

				#determine the current points and the neighboring points
#				print(qid, Q)
#				print(newQ)
#				print(qid, len(Q), len(newQ))
				currentScore, currentMask = Q[qid]
				minDims = point_voxels[currentMask, :].min(axis=0)
				maxDims = point_voxels[currentMask, :].max(axis=0)
				if qid==0:
					bestMask = currentMask
					if not numpy.any(minDims<seqMinDims) and not numpy.any(maxDims>seqMaxDims):
						if stuck >= 1:
#							print('stuck')
							break
						else:
							stuck += 1
					else:
						stuck = 0
					seqMinDims = numpy.minimum(seqMinDims, minDims)
					seqMaxDims = numpy.maximum(seqMaxDims, maxDims)
#				if currentScore > bestScore:
#					bestScore = currentScore
#					bestMask = currentMask
				currentPoints = points[currentMask, :]
				newMinDims = minDims.copy()	
				newMaxDims = maxDims.copy()	
				newMinDims -= 1
				newMaxDims += 1
				mask = numpy.logical_and(numpy.all(point_voxels>=newMinDims,axis=1), numpy.all(point_voxels<=newMaxDims, axis=1))
				mask = numpy.logical_and(mask, numpy.logical_not(currentMask))
				mask = numpy.logical_and(mask, numpy.logical_not(visited))
				expandPoints = points[mask, :]
				expandClass = obj_id[mask] == target_id
				rejectClass = obj_id[currentMask] != target_id
				
				if len(expandPoints) > 0:
					#randomly generate several search candidates from currentMask
					for search_id in range(SEARCH_WIDTH):

						if len(currentPoints) >= NUM_INLIER_POINT:
							subset = numpy.random.choice(len(currentPoints), NUM_INLIER_POINT, replace=False)
						else:
							subset = range(len(currentPoints)) + list(numpy.random.choice(len(currentPoints), NUM_INLIER_POINT-len(currentPoints), replace=True))
						center = numpy.median(currentPoints, axis=0)
						ep = numpy.array(expandPoints)
						ep[:,:2] -= center[:2]
						ep[:,6:] -= center[6:]
						inlier_points[0,:,:] = numpy.array(currentPoints[subset, :])
						inlier_points[0,:,:2] -= center[:2]
						inlier_points[0,:,6:] -= center[6:]
						input_remove[0,:] = numpy.array(rejectClass)[subset]
						if len(ep) >= NUM_NEIGHBOR_POINT:
							subset = numpy.random.choice(len(ep), NUM_NEIGHBOR_POINT, replace=False)
						else:
							subset = range(len(ep)) + list(numpy.random.choice(len(ep), NUM_NEIGHBOR_POINT-len(ep), replace=True))
						neighbor_points[0,:,:] = numpy.array(ep)[subset, :]
						input_add[0,:] = numpy.array(expandClass)[subset]
						add,add_acc, rmv,rmv_acc = sess.run([net.add_output, net.add_acc, net.remove_output, net.remove_acc],
							{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})

						add_conf = scipy.special.softmax(add[0], axis=-1)[:,1]
						rmv_conf = scipy.special.softmax(rmv[0], axis=-1)[:,1]
						add_mask = numpy.random.random(len(add_conf)) < add_conf
						rmv_mask = numpy.random.random(len(rmv_conf)) < rmv_conf
						addPoints = neighbor_points[0,:,:][add_mask]
						addPoints[:,:2] += center[:2]
						addVoxels = numpy.round(addPoints[:,:3]/resolution).astype(int)
						addSet = set([tuple(p) for p in addVoxels])
						addLogProb = 0
						for i in range(len(neighbor_points[0])):
							neighbor_points[0,i,:2] += center[:2]
							p = tuple(numpy.round(neighbor_points[0,i,:3]/resolution).astype(int))
							if p in addSet:
								addLogProb += numpy.log(add_conf[i]) / NUM_NEIGHBOR_POINT
							else:
								addLogProb += numpy.log((1 - add_conf[i])) / NUM_NEIGHBOR_POINT
						rmvPoints = inlier_points[0,:,:][rmv_mask]
						rmvPoints[:,:2] += center[:2]
						rmvVoxels = numpy.round(rmvPoints[:,:3]/resolution).astype(int)
						rmvSet = set([tuple(p) for p in rmvVoxels])
						rmvLogProb = 0
						for i in range(len(inlier_points[0])):
							inlier_points[0,i,:2] += center[:2]
							p = tuple(numpy.round(inlier_points[0,i,:3]/resolution).astype(int))
							if p in rmvSet:
								rmvLogProb += numpy.log(rmv_conf[i]) / NUM_NEIGHBOR_POINT
							else:
								rmvLogProb += numpy.log((1 - rmv_conf[i])) / NUM_NEIGHBOR_POINT
						updated = False

						newMask = currentMask.copy()
						for i in range(len(point_voxels)):
							if not newMask[i] and tuple(point_voxels[i]) in addSet:
								newMask[i] = True
								updated = True
							if tuple(point_voxels[i]) in rmvSet:
								newMask[i] = False
						iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,newMask)) / numpy.sum(numpy.logical_or(gt_mask,newMask))
#						print('%d/%d points %d outliers %d add %d rmv %.2f iou'%(numpy.sum(numpy.logical_and(newMask, gt_mask)), numpy.sum(gt_mask),
#							numpy.sum(numpy.logical_and(gt_mask==0, newMask)), len(addSet), len(rmvSet), iou))
						steps += 1
						if updated:
							if scoring=='ml':
								newScore = currentScore + addLogProb + rmvLogProb
							elif scoring=='np':
								newScore = numpy.sum(newMask)
							newQ.append((newScore, newMask))

				if qid < len(Q) - 1:
					qid += 1
				else:
					qid = 0
					Q = sorted(newQ, key=lambda x:x[0], reverse=True)[:BEAM_WIDTH]
					newQ = []

			visited[bestMask] = True
			if numpy.sum(bestMask) > cluster_threshold:
				cluster_label[bestMask] = cluster_id
				cluster_id += 1
				iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,bestMask)) / numpy.sum(numpy.logical_or(gt_mask,bestMask))
				print('room %d target %3d %.4s: step %3d %4d/%4d points IOU %.3f add %.3f rmv %.3f'%(room_id, target_id, target_class, steps, numpy.sum(bestMask), numpy.sum(gt_mask), iou, add_acc, rmv_acc))

		#fill in points with no labels
		nonzero_idx = numpy.nonzero(cluster_label)[0]
		nonzero_points = points[nonzero_idx, :]
		filled_cluster_label = cluster_label.copy()
		for i in numpy.nonzero(cluster_label==0)[0]:
			d = numpy.sum((nonzero_points - points[i])**2, axis=1)
			closest_idx = numpy.argmin(d)
			filled_cluster_label[i] = cluster_label[nonzero_idx[closest_idx]]
		cluster_label = filled_cluster_label

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
		if AREA.isdigit():
			print(room_name[room_id])
		print("Area %s room %d NMI: %.2f AMI: %.2f ARS: %.2f PRC: %.2f RCL: %.2f IOU: %.2f"%(str(AREA), room_id, nmi,ami,ars, prc, rcl, room_iou))

		#save point cloud results to file
		if save_results:
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			obj_color[0] = [100,100,100]
			unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
			savePLY('data/results/%d.ply'%save_id, unequalized_points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))

