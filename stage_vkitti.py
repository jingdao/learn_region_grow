import numpy
import h5py
import glob
import sys
from learn_region_grow_util import savePCD
import networkx as nx
import itertools

max_x_range = 10
max_y_range = 10
downsample_resolution = 0.05
cluster_resolution = 0.2
stacked_points = []
stacked_count = []

for AREA in range(1,7):
#for AREA in [1]:
	for filename in glob.glob('data/vkitti3d_dataset_v1.0/0%d/*.npy'%AREA):
		print('Processing',filename)
		pcd = numpy.load(filename)
		class_id = pcd[:,-1].astype(int)

#		mask = numpy.logical_and(pcd[:,0]<max_x_range, numpy.abs(pcd[:,1])<max_y_range)
#		pcd = pcd[mask]
#		class_id = class_id[mask]

		equalized_idx = []
		equalized_map = {}
		for i in range(len(pcd)):
			k = tuple(numpy.round(pcd[i,:3]/downsample_resolution).astype(int))
			if not k in equalized_map:
				equalized_map[k] = len(equalized_idx)
				equalized_idx.append(i)
		pcd = pcd[equalized_idx, :]
		class_id = class_id[equalized_idx]

		neighbor_map = {}
		point_voxels = numpy.round(pcd[:,:3]/cluster_resolution).astype(int)
		for i in range(len(pcd)):
			k = tuple(point_voxels[i])
			if not k in neighbor_map:
				neighbor_map[k] = []
			neighbor_map[k].append(i)

		remove_set = set()
		for k in neighbor_map:
			if len(neighbor_map[k]) < 3:
				remove_set.add(k)
		clean_idx = []
		for i in range(len(pcd)):
			k = tuple(point_voxels[i])
			if not k in remove_set:
				clean_idx.append(i)
		pcd = pcd[clean_idx, :]
		class_id = class_id[clean_idx]
		point_voxels = numpy.round(pcd[:,:3]/cluster_resolution).astype(int)
		neighbor_map = {}
		for i in range(len(pcd)):
			k = tuple(point_voxels[i])
			if not k in neighbor_map:
				neighbor_map[k] = []
			neighbor_map[k].append(i)

		cluster_label = numpy.zeros(len(pcd),dtype=int)
		cluster_id = 1
		for i in range(len(pcd)):
			if cluster_label[i] > 0:
				continue
			visited = set()
			k = tuple(point_voxels[i])
			c = class_id[i]
			Q = [k]
			while len(Q) > 0:
				q = Q[-1]
				del Q[-1]
				visited.add(q)
				matched = False
				for j in neighbor_map[q]:
					if class_id[j]==c:
						cluster_label[j] = cluster_id
						matched = True
				if matched:
					for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
						kk = (q[0]+offset[0], q[1]+offset[1], q[2]+offset[2])
						if kk in neighbor_map and not kk in visited:
							Q.append(kk)
			cluster_id += 1

		min_cluster_size = 50
		new_id = 1
		for i in range(cluster_id):
			mask = cluster_label==i
			if numpy.sum(mask) >= min_cluster_size:
				cluster_label[mask] = new_id
				new_id += 1
			else:
				cluster_label[mask] = 0
		mask = cluster_label > 0
		pcd = pcd[mask]
		cluster_label = cluster_label[mask]
		class_id = class_id[mask]
		print("%d points %d clusters"%(len(pcd), new_id))

#		obj_color = numpy.random.randint(0,255,(numpy.max(cluster_label)+1,3))
#		obj_color[0] = [0,0,0]
#		pcd[:,3:6] = obj_color[cluster_label,:]
#		savePCD('tmp/0-cloud.pcd', pcd)
#		for i in range(cluster_label.max()+1):
#			p = pcd[cluster_label==i, :]
#			if len(p) > 0:
#				savePCD('tmp/%d-cloud.pcd'%i, p)
#		sys.exit(1)

		points = numpy.zeros((len(pcd), 8), dtype=numpy.float32)
		points[:,:3] = pcd[:,:3]
		points[:,3:6] = pcd[:,3:6]/255.0 - 0.5
		points[:,6] = cluster_label
		points[:,-1] = class_id
		stacked_points.append(points)
		stacked_count.append(len(pcd))
		break

output_filename = 'data/vkitti.h5'
h5_fout = h5py.File(output_filename,'w')
h5_fout.create_dataset( 'points', data=numpy.vstack(stacked_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
h5_fout.create_dataset( 'count_room', data=stacked_count, compression='gzip', compression_opts=4, dtype=numpy.int32)
h5_fout.close()

