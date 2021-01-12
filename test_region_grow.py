import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from class_util import classes_s3dis, classes_nyu40, classes_kitti, class_to_id, class_to_color_rgb
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

numpy.random.seed(0)
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
FEATURE_SIZE = 13
LITE = None
TEST_AREAS = ['1','2','3','4','5','6','scannet']
resolution = 0.1
add_threshold = 0.5
rmv_threshold = 0.5
cluster_threshold = 10
save_results = False
cross_domain = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []
comp_time_analysis = {
	'feature': [],
	'net': [],
	'neighbor': [],
	'inlier': [],
	'current_net' : [],
	'current_neighbor' : [],
	'current_inlier' : [],
	'iter_net' : [],
	'iter_neighbor' : [],
	'iter_inlier' : [],
}

for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		TEST_AREAS = sys.argv[i+1].split(',')
	elif sys.argv[i]=='--save':
		save_results = True
	elif sys.argv[i]=='--cross-domain':
		cross_domain = True
	elif sys.argv[i]=='--train-area':
		TRAIN_AREA = sys.argv[i+1]
	elif sys.argv[i]=='--resolution':
		resolution = float(sys.argv[i+1])
	elif sys.argv[i]=='--lite':
		LITE = int(sys.argv[i+1])

for AREA in TEST_AREAS:
	tf.reset_default_graph()
	if cross_domain:
		MODEL_PATH = 'models/cross_domain/lrgnet_%s.ckpt' % TRAIN_AREA
	else:
		if FEATURE_SIZE==6:
			MODEL_PATH = 'models/lrgnet_model%s_xyz.ckpt'%AREA
		elif FEATURE_SIZE==9:
			MODEL_PATH = 'models/lrgnet_model%s_xyzrgb.ckpt'%AREA
		elif FEATURE_SIZE==12:
			MODEL_PATH = 'models/lrgnet_model%s_xyzrgbn.ckpt'%AREA
		else:
			# use full set of features
			if NUM_INLIER_POINT!=512 or NUM_NEIGHBOR_POINT!=512:
				MODEL_PATH = 'models/lrgnet_model%s_i_%d_j_%d.ckpt'%(AREA, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT)
			elif LITE is not None:
				MODEL_PATH = 'models/lrgnet_model%s_lite_%d.ckpt'%(AREA, LITE)
			else:
				MODEL_PATH = 'models/lrgnet_model%s.ckpt'%AREA
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)
	net = LrgNet(1, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE, LITE)
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_PATH)
	print('Restored from %s'%MODEL_PATH)

	if AREA in ['scannet', 's3dis', 'kitti_train', 'kitti_val']:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/%s.h5' % AREA)
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)
	classes = classes_kitti if 'kitti' in AREA else classes_nyu40 if AREA=='scannet' else classes_s3dis
	room_name_file = 'data/%s_room_name.txt' % AREA
	if os.path.exists(room_name_file):
		room_names = open(room_name_file, 'r').read().split('\n')
	else:
		room_names = None
	sample_file = open('/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/jsis3d/data/s3dis/metadata/s3dis_sampled.txt', 'r')
	sample_list = set(sample_file.read().split('\n'))
	sample_file.close()

	for room_id in range(len(all_points)):
#	for room_id in [162, 157, 166, 169, 200]:
#	for room_id in [10, 44, 87, 111, 198]:
		if room_names is not None and not '_'.join(room_names[room_id].split())+'.h5' in sample_list:
			continue
		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]

		#equalize resolution
		t1 = time.time()
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
		if FEATURE_SIZE==6:
			points = numpy.hstack((xyz, room_coordinates)).astype(numpy.float32)
		elif FEATURE_SIZE==9:
			points = numpy.hstack((xyz, room_coordinates, rgb)).astype(numpy.float32)
		elif FEATURE_SIZE==12:
			points = numpy.hstack((xyz, room_coordinates, rgb, normals)).astype(numpy.float32)
		else:
			points = numpy.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(numpy.float32)
		comp_time_analysis['feature'].append(time.time() - t1)

		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		cluster_label = numpy.zeros(len(points), dtype=int)
		cluster_id = 1
		visited = numpy.zeros(len(point_voxels), dtype=bool)
		inlier_points = numpy.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=numpy.float32)
		neighbor_points = numpy.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=numpy.float32)
		input_add = numpy.zeros((1, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
		input_remove = numpy.zeros((1, NUM_INLIER_POINT), dtype=numpy.int32)
		order = numpy.argsort(curvatures)
		#iterate over each object in the room
#		for seed_id in range(len(point_voxels)):
		for seed_id in numpy.arange(len(points))[order]:
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
			maskLogProb = 0

			#perform region growing
			while True:

				def stop_growing(reason):
					global cluster_id, start_time
					visited[currentMask] = True
					if numpy.sum(currentMask) > cluster_threshold:
						cluster_label[currentMask] = cluster_id
						cluster_id += 1
						iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
						print('room %d target %3d %.4s: step %3d %4d/%4d points IOU %.3f add %.3f rmv %.3f %s'%(room_id, target_id, target_class, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, add_acc, rmv_acc, reason))

				#determine the current points and the neighboring points
				t = time.time()
				currentPoints = points[currentMask, :].copy()
				newMinDims = minDims.copy()	
				newMaxDims = maxDims.copy()	
				newMinDims -= 1
				newMaxDims += 1
				mask = numpy.logical_and(numpy.all(point_voxels>=newMinDims,axis=1), numpy.all(point_voxels<=newMaxDims, axis=1))
				mask = numpy.logical_and(mask, numpy.logical_not(currentMask))
				mask = numpy.logical_and(mask, numpy.logical_not(visited))
				expandPoints = points[mask, :].copy()
				expandClass = obj_id[mask] == target_id
				rejectClass = obj_id[currentMask] != target_id
				
				if len(expandPoints)==0: #no neighbors (early termination)
					stop_growing('noneighbor')
					break

				if len(currentPoints) >= NUM_INLIER_POINT:
					subset = numpy.random.choice(len(currentPoints), NUM_INLIER_POINT, replace=False)
				else:
					subset = list(range(len(currentPoints))) + list(numpy.random.choice(len(currentPoints), NUM_INLIER_POINT-len(currentPoints), replace=True))
				center = numpy.median(currentPoints, axis=0)
				expandPoints = numpy.array(expandPoints)
				expandPoints[:,:2] -= center[:2]
				expandPoints[:,6:] -= center[6:]
				inlier_points[0,:,:] = currentPoints[subset, :]
				inlier_points[0,:,:2] -= center[:2]
				inlier_points[0,:,6:] -= center[6:]
				input_remove[0,:] = numpy.array(rejectClass)[subset]
				if len(expandPoints) >= NUM_NEIGHBOR_POINT:
					subset = numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = list(range(len(expandPoints))) + list(numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
				neighbor_points[0,:,:] = numpy.array(expandPoints)[subset, :]
				input_add[0,:] = numpy.array(expandClass)[subset]
				comp_time_analysis['current_neighbor'].append(time.time() - t)
				t = time.time()
				ls, add,add_acc, rmv,rmv_acc = sess.run([net.loss, net.add_output, net.add_acc, net.remove_output, net.remove_acc],
					{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})
				comp_time_analysis['current_net'].append(time.time() - t)
				t = time.time()

				add_conf = scipy.special.softmax(add[0], axis=-1)[:,1]
				rmv_conf = scipy.special.softmax(rmv[0], axis=-1)[:,1]
#				add_mask = add_conf > add_threshold
#				rmv_mask = rmv_conf > rmv_threshold
				add_mask = numpy.random.random(len(add_conf)) < add_conf
				rmv_mask = numpy.random.random(len(rmv_conf)) < rmv_conf
#				add_mask = input_add[0].astype(bool)
#				rmv_mask = input_remove[0].astype(bool)
				addPoints = neighbor_points[0,:,:][add_mask]
				addPoints[:,:2] += center[:2]
				addVoxels = numpy.round(addPoints[:,:3]/resolution).astype(int)
				addSet = set([tuple(p) for p in addVoxels])
				rmvPoints = inlier_points[0,:,:][rmv_mask]
				rmvPoints[:,:2] += center[:2]
				rmvVoxels = numpy.round(rmvPoints[:,:3]/resolution).astype(int)
				rmvSet = set([tuple(p) for p in rmvVoxels])
				updated = False
				iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
#				print('%d/%d points %d outliers %d/%d add %d/%d rmv %.2f iou'%(numpy.sum(numpy.logical_and(currentMask, gt_mask)), numpy.sum(gt_mask),
#					numpy.sum(numpy.logical_and(gt_mask==0, currentMask)), len(addSet), len(expandPoints), len(rmvSet), len(currentPoints), iou))
				for i in range(len(point_voxels)):
					if not currentMask[i] and tuple(point_voxels[i]) in addSet:
						currentMask[i] = True
						updated = True
					if tuple(point_voxels[i]) in rmvSet:
						currentMask[i] = False
				steps += 1
				comp_time_analysis['current_inlier'].append(time.time() - t)

				if updated: #continue growing
					minDims = point_voxels[currentMask, :].min(axis=0)
					maxDims = point_voxels[currentMask, :].max(axis=0)
					if not numpy.any(minDims<seqMinDims) and not numpy.any(maxDims>seqMaxDims):
						if stuck >= 1:
							stop_growing('stuck')
							break
						else:
							stuck += 1
					else:
						stuck = 0
					seqMinDims = numpy.minimum(seqMinDims, minDims)
					seqMaxDims = numpy.maximum(seqMaxDims, maxDims)
				else: #no matching neighbors (early termination)
					stop_growing('noexpand')
					break

		#fill in points with no labels
		nonzero_idx = numpy.nonzero(cluster_label)[0]
		nonzero_points = points[nonzero_idx, :]
		filled_cluster_label = cluster_label.copy()
		for i in numpy.nonzero(cluster_label==0)[0]:
			d = numpy.sum((nonzero_points - points[i])**2, axis=1)
			closest_idx = numpy.argmin(d)
			filled_cluster_label[i] = cluster_label[nonzero_idx[closest_idx]]
		cluster_label = filled_cluster_label
		print('%s %d points: %.2fs' % (room_names[room_id] if room_names is not None else '', len(unequalized_points), time.time() - t1))

		#calculate statistics 
		gt_match = 0
		match_id = 0
		dt_match = numpy.zeros(cluster_label.max(), dtype=bool)
		cluster_label2 = numpy.zeros(len(cluster_label), dtype=int)
		room_iou = []
		unique_id, count = numpy.unique(obj_id, return_counts=True)
		for k in range(len(unique_id)):
			i = unique_id[numpy.argsort(count)][::-1][k]
			best_iou = 0
			for j in range(1, cluster_label.max()+1):
				if not dt_match[j-1]:
					iou = 1.0 * numpy.sum(numpy.logical_and(obj_id==i, cluster_label==j)) / numpy.sum(numpy.logical_or(obj_id==i, cluster_label==j))
					best_iou = max(best_iou, iou)
					if iou > 0.5:
						dt_match[j-1] = True
						gt_match += 1
						cluster_label2[cluster_label==j] = k+1
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

		comp_time_analysis['neighbor'].append(sum(comp_time_analysis['current_neighbor']))
		comp_time_analysis['iter_neighbor'].extend(comp_time_analysis['current_neighbor'])
		comp_time_analysis['current_neighbor'] = []
		comp_time_analysis['net'].append(sum(comp_time_analysis['current_net']))
		comp_time_analysis['iter_net'].extend(comp_time_analysis['current_net'])
		comp_time_analysis['current_net'] = []
		comp_time_analysis['inlier'].append(sum(comp_time_analysis['current_inlier']))
		comp_time_analysis['iter_inlier'].extend(comp_time_analysis['current_inlier'])
		comp_time_analysis['current_inlier'] = []

		#save point cloud results to file
		if save_results:
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			obj_color[0] = [100,100,100]
			unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
			if AREA == 'scannet':
				savePLY('data/results/lrg/scannet%d.ply'%save_id, unequalized_points)
			else:
				savePLY('data/results/lrg/%d.ply'%save_id, unequalized_points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))
total_time = 0
for i in list(comp_time_analysis.keys()):
	if not i.startswith('current'):
		comp_time_analysis['std_'+i] = numpy.std(comp_time_analysis[i])
		comp_time_analysis[i] = numpy.mean(comp_time_analysis[i])
		total_time += comp_time_analysis[i]
for i in comp_time_analysis:
	if not i.startswith('current') and not i.startswith('std'):
		print('%10s %6.2f+-%5.2fs %4.1f' % (i, comp_time_analysis[i], comp_time_analysis['std_'+i], 100.0 * comp_time_analysis[i] / total_time))
