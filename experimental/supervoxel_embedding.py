import numpy
import h5py
import sys
from class_util import classes, class_to_id, class_to_color_rgb
import itertools
import random
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import math
import networkx as nx
import time
from learn_region_grow_util import *

numpy.random.seed(0)
resolution = 0.1
supervoxel_resolution = 0.5
save_id = 0

#for AREA in range(1,7):
for AREA in [3]:
	all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
	stacked_points = []
	stacked_neighbor_points = []
	stacked_labels = []

#	for room_id in range(len(all_points)):
	for room_id in [0,1,2]:
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
		points = unequalized_points[equalized_idx] #(N,6)
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]

		#compute normals and curvatures
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
			curvatures.append(numpy.fabs(curvature)) # change to absolute values?
		normals = numpy.array(normals) #(N,3)
		curvatures = numpy.array(curvatures) #(N,)
		points = numpy.hstack((points, normals, curvatures.reshape(-1,1))).astype(numpy.float32)

		#compute supervoxel seeds
		supervoxel_set = set()
		supervoxel_idx = []
		for i in range(len(points)):
			k = tuple(numpy.round(points[i,:3]/supervoxel_resolution).astype(int))
			if not k in supervoxel_set:
				supervoxel_set.add(k)
				supervoxel_idx.append(i)
		supervoxel_features = points[supervoxel_idx, :]
		print(len(supervoxel_idx),'supervoxels')

		#compute supervoxel membership
		cluster_label = numpy.zeros(len(points),dtype=int)
		for i in range(len(points)):
			closest_idx = numpy.argmin(numpy.sum((points[i,:] - supervoxel_features)**2, axis=1))
			cluster_label[i] = closest_idx

		obj_color = numpy.random.randint(0,255,(numpy.max(cluster_label)+1,3))
		points[:,3:6] = obj_color[cluster_label,:]
		savePCD('tmp/%d-cloud.pcd'%save_id, points)
		print('Saved %d clusters to %d-cloud.pcd'%(len(supervoxel_set), save_id))
		save_id += 1
