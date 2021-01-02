import numpy
import h5py
import os
import sys
from class_util import classes_s3dis, classes_nyu40, classes_kitti, class_to_id, class_to_color_rgb
import itertools
import random
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.externals import joblib
import math
import networkx as nx
from scipy.cluster.vq import vq, kmeans
import time
import matplotlib.pyplot as plt
import scipy.special

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
property uchar red
property uchar green
property uchar blue
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

def loadFPFH(filename):
	pcd = open(filename,'r')
	for l in pcd:
		if l.startswith('DATA'):
			break
	features = []
	for l in pcd:
		features.append([float(t) for t in l.split()[:33]])
	features = numpy.array(features)
	return features

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb normal_x normal_y normal_z curvature
SIZE 4 4 4 4 4 4 4 4
TYPE F F F I F F F F
COUNT 1 1 1 1 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d %f %f %f %f\n"%(p[0],p[1],p[2],rgb,p[6],p[7],p[8],p[9]))
	f.close()
	print('Saved %d points to %s' % (l,filename))

numpy.random.seed(0)
TRAIN_AREA = 's3dis'
TEST_AREAS = [1,2,3,4,5,6,'scannet']
resolution = 0.1
feature_size = 9
NUM_POINT = 1024
mode = 'normal'
threshold = None
save_results = False
cross_domain = False
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
	elif sys.argv[i]=='--train-area':
		TRAIN_AREA = sys.argv[i+1]
	elif sys.argv[i]=='--threshold':
		threshold = float(sys.argv[i+1])
	elif sys.argv[i]=='--resolution':
		resolution = float(sys.argv[i+1])
	elif sys.argv[i]=='--save':
		save_results = True
	elif sys.argv[i]=='--cross-domain':
		cross_domain = True

if threshold is None:
    if mode=='normal':
        threshold = 0.99
    elif mode == 'curvature':
        threshold = 0.01
    elif mode=='color':
        threshold = 0.005
    elif mode=='smoothness':
        threshold = 0.985 if TEST_AREAS[0]=='scannet' else 0.98
    elif mode=='fpfh':
        threshold = 0.985
    elif mode=='feature':
        threshold = 0.98
        threshold2 = 0.1
        threshold3 = 0.1
    else:
        threshold = 0.99
print('Using threshold', threshold, 'resolution',resolution)
NUM_CLASSES = len(classes_kitti) if 'kitti' in TRAIN_AREA else len(classes_nyu40) if 'scannet' in TRAIN_AREA else len(classes_s3dis)

if mode in ['pointnet', 'pointnet2']:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from train_pointnet import PointNet, PointNet2

for AREA in TEST_AREAS:
	if mode in ['pointnet', 'pointnet2']:
		tf.reset_default_graph()
		if cross_domain:
			MODEL_PATH = 'models/cross_domain/%s_%s.ckpt' % (mode, TRAIN_AREA)
		else:
			if AREA == 'scannet':
				MODEL_PATH = 'models/%s_model5.ckpt' % mode
			else:
				MODEL_PATH = 'models/%s_model%s.ckpt' % (mode, AREA)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)
		if mode=='pointnet':
			net = PointNet(1,NUM_POINT,NUM_CLASSES) 
		else:
			net = PointNet2(1,NUM_POINT,NUM_CLASSES) 
		saver = tf.train.Saver()
		saver.restore(sess, MODEL_PATH)
		print('Restored from %s'%MODEL_PATH)
	elif mode=='edge':
		if AREA == 'scannet':
			MODEL_PATH = 'models/edge5.pkl'
		else:
			MODEL_PATH = 'models/edge%s.pkl' % AREA
		svc = joblib.load(MODEL_PATH)
		print('Restored from %s'%MODEL_PATH)

	if AREA in ['scannet', 's3dis', 'kitti_train', 'kitti_val']:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/%s.h5' % AREA)
	else:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)

	for room_id in range(len(all_points)):
#	for room_id in [162, 157, 166, 169, 200]:
#	for room_id in [10, 44, 87, 111, 198]:
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

		#compute normals
		if mode=='normal' or mode=='curvature' or mode=='smoothness' or mode=='fpfh' or mode=='feature':
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
			if mode == 'normal':
				points = numpy.hstack((points, normals)).astype(numpy.float32) #(N, 9)
			if mode == 'curvature':
				points = numpy.hstack((points,numpy.reshape(curvatures,(curvatures.shape[0],1)))).astype(numpy.float32) #(N, 7)
			if mode == 'fpfh':
				points = numpy.hstack((points, normals, curvatures.reshape(-1, 1))).astype(numpy.float32) #(N, 10)


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
						if kk in voxel_map and abs(curvatures[voxel_map[kk]] - curvatures[i]) < threshold:
							edges.append([i, voxel_map[kk]])
		elif mode=='color':
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and numpy.sum((points[voxel_map[kk],3:6] - points[i,3:6])**2) < threshold:
							edges.append([i, voxel_map[kk]])
		elif mode=='pointnet' or mode=='pointnet2':
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

		elif mode=='edge':
			def get_features(E, p1, p2, neighbor_min, neighbor_max, neighbor_mean):
				F = numpy.hstack((
					0.5 * (p1[:,2:] + p2[:,2:]),
					numpy.minimum(p1[:,2:], p2[:,2:]),
					numpy.maximum(p1[:,2:], p2[:,2:]),
					numpy.abs(p1 - p2),
					numpy.maximum(
						numpy.abs(p1 - neighbor_min[E[:,1]]),
						numpy.abs(p2 - neighbor_min[E[:,0]]),
					),
					numpy.maximum(
						numpy.abs(p1 - neighbor_max[E[:,1]]),
						numpy.abs(p2 - neighbor_max[E[:,0]]),
					),
				))
				return F
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map:
							edges.append([i, voxel_map[kk]])
			E = numpy.array(edges)
			neighbor_array = [[i] for i in range(len(points))]
			for e in E:
				neighbor_array[int(e[0])].append(int(e[1]))
				neighbor_array[int(e[1])].append(int(e[0]))
			neighbor_points = [points[e,:6] for e in neighbor_array]
			neighbor_min = numpy.array([n.min(axis=0) for n in neighbor_points])
			neighbor_max = numpy.array([n.max(axis=0) for n in neighbor_points])
			neighbor_mean = numpy.array([n.mean(axis=0) for n in neighbor_points])
			p1 = points[E[:,0],:6]
			p2 = points[E[:,1],:6]
			F = get_features(E, p1, p2, neighbor_min, neighbor_max, neighbor_mean)

			test_probs = svc.predict_proba(F)[:,1]
			neighbors = [[0] for i in range(len(points))]
			for i in range(len(E)):
				neighbors[int(E[i,0])].append(test_probs[i])
				neighbors[int(E[i,1])].append(test_probs[i])
			neighbor_max = numpy.array([numpy.max(n) for n in neighbors])
			criteria = numpy.logical_and(test_probs > 0.99 * neighbor_max[E[:,0]], test_probs > 0.99 * neighbor_max[E[:,1]])
			criteria = numpy.logical_and(criteria, test_probs > 0.9)
			edges = E[criteria].tolist()
		elif mode=='fpfh':
			savePCD('data/tmp.pcd', points)
			os.system('pcl_fpfh_estimation data/tmp.pcd data/fpfh.pcd -radius %f' % (resolution*2))
			os.system('pcl_convert_pcd_ascii_binary data/fpfh.pcd data/fpfh_ascii.pcd 0')
			fpfh = loadFPFH('data/fpfh_ascii.pcd')
			normalizer = numpy.tile(numpy.linalg.norm(fpfh,axis=1).reshape(-1, 1), [1, fpfh.shape[1]])	
			fpfh /= normalizer
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and fpfh[voxel_map[kk]].dot(fpfh[i]) > threshold:
							edges.append([i, voxel_map[kk]])
		elif mode=='feature':
			for i in range(len(point_voxels)):
				k = tuple(point_voxels[i])
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in voxel_map and \
								normals[voxel_map[kk]].dot(normals[i]) > threshold and \
								abs(curvatures[voxel_map[kk]] - curvatures[i]) < threshold2 and \
								numpy.sum((points[voxel_map[kk],3:6] - points[i,3:6])**2) < threshold3:
							edges.append([i, voxel_map[kk]])

		if mode=='smoothness':
			#use smoothness constraint for region growing (Rabbani et al.)
			cluster_label = numpy.zeros(len(point_voxels), dtype=int)
			visited = numpy.zeros(len(point_voxels), dtype=bool)
			cluster_id = 1
			min_cluster_size = 10
			for seed_id in numpy.arange(len(point_voxels))[numpy.argsort(curvatures)]:
				if visited[seed_id]:
					continue
				Q = [seed_id]
				C = []
				while len(Q) > 0:
					i = Q[-1]
					del Q[-1]
					C.append(i)
					visited[i] = True
					k = tuple(point_voxels[i])
					for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
						if offset!=(0,0,0):
							kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
							if kk in voxel_map and not visited[voxel_map[kk]] and normals[voxel_map[kk]].dot(normals[i]) > threshold:
								Q.append(voxel_map[kk])
				if len(C) > min_cluster_size:
					cluster_label[C] = cluster_id
					cluster_id += 1
		else:
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

			if mode=='edge':
				best_neighbor = [[] for i in range(len(points))]
				for i in range(len(E)):
					best_neighbor[E[i,0]].append([E[i,1], test_probs[i]])
					best_neighbor[E[i,1]].append([E[i,0], test_probs[i]])
				best_neighbor = [sorted(b,key=lambda x:x[1]) for b in best_neighbor]
				for i in numpy.nonzero(cluster_label==0)[0]:
					visited = set()
					Q = [[i,1]]
					while len(Q) > 0:
						q = Q[-1][0]
						del Q[-1]
						if q in visited:
							continue
						if cluster_label[q] > 0:
							cluster_label[i] = cluster_label[q]
							break
						visited.add(q)
						Q.extend(best_neighbor[q])
	
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

		#save point cloud results to file
		if save_results:
			if mode=='normal':
				unequalized_points[:,3:6] = normals[unequalized_idx]*255
				savePLY('data/normal/%d.ply'%save_id, unequalized_points)
			elif mode == 'curvature':
				curvatures = curvatures / curvatures.max()
				jet = plt.get_cmap('jet')
				color_map = jet(curvatures)[:,:3] * 255
				unequalized_points[:,3:6] = color_map[unequalized_idx]
				savePLY('data/curvature/%d.ply'%save_id, unequalized_points)
			elif mode=='pointnet' or mode=='pointnet2':
				unequalized_points[:,3:6] = [class_to_color_rgb[c] for c in class_labels[unequalized_idx]]
				savePLY('data/class/%d.ply'%save_id, unequalized_points)
			color_sample_state = numpy.random.RandomState(0)
			obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
			obj_color[0] = [100,100,100]
			unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
			if AREA == 'scannet':
				savePLY('data/results/%s/scannet%d.ply'%(mode,save_id), unequalized_points)
			else:
				savePLY('data/results/%s/%d.ply'%(mode,save_id), unequalized_points)
			save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
	(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
	numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))

