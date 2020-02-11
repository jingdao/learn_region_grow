from learn_region_grow_util import *
import itertools
import sys
from class_util import classes
from sklearn.decomposition import PCA

resolution = 0.1
repeats_per_room = 1
local_range = 2
resolution = 0.1
batch_size = 1
num_neighbors = 50
neighbor_radii = 0.3
hidden_size = 200
embedding_size = 10
feature_size = 6
save_id = 0

SEED = None
numpy.random.seed(0)
for i in range(len(sys.argv)):
	if sys.argv[i]=='--seed':
		SEED = int(sys.argv[i+1])
		numpy.random.seed(SEED)

for AREA in range(1,7):
#for AREA in [3]:

	tf.reset_default_graph()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)
	net = MCPNet(batch_size, num_neighbors, feature_size, hidden_size, embedding_size)
	saver = tf.train.Saver()
	MODEL_PATH = 'models/mcpnet_model%s.ckpt'%AREA
	saver = tf.train.Saver()
	saver.restore(sess, MODEL_PATH)
	print('Restored from %s'%MODEL_PATH)

	all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
	stacked_points = []
	stacked_neighbor_points = []
	stacked_count = []
	stacked_neighbor_count = []
	stacked_class = []
	stacked_steps = []
	stacked_complete = []

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
		normal_grid = {}
		for i in range(len(unequalized_points)):
			k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_map:
				equalized_map[k] = len(equalized_idx)
				equalized_idx.append(i)
				kk = tuple(numpy.round(unequalized_points[i,:3]/neighbor_radii).astype(int))
				if not kk in coarse_map:
					coarse_map[kk] = []
				coarse_map[kk].append(equalized_map[k])
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
			unequalized_idx.append(equalized_map[k])
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]

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
			curvatures.append(numpy.fabs(curvature)) # change to absolute values?
		normals = numpy.array(normals)
		curvatures = numpy.array(curvatures)
		points = numpy.hstack((points, normals)).astype(numpy.float32)

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
		input_points = numpy.zeros((batch_size, feature_size-2), dtype=float)
		input_neighbors = numpy.zeros((batch_size, num_neighbors, feature_size), dtype=float)
		num_batches = 0
		for i in range(len(points)):
			input_points[0,:] = points[i, 2:6]	
			input_neighbors[0,:,:] = neighbor_array[i, :, :feature_size]
			emb_val = sess.run(net.embeddings, {net.input_pl:input_points, net.neighbor_pl:input_neighbors})
			embeddings[i] = emb_val
			num_batches += 1

		points = numpy.hstack((points, embeddings)).astype(numpy.float32)
		point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
		for i in range(repeats_per_room):
			visited = numpy.zeros(len(point_voxels), dtype=bool)
			#iterate over each voxel in the room
			for seed_id in numpy.random.choice(range(len(points)), len(points), replace=False):
				if visited[seed_id]:
					continue
				target_id = obj_id[seed_id]
				obj_voxels = point_voxels[obj_id==target_id, :]
				gt_mask = obj_id==target_id
				original_minDims = obj_voxels.min(axis=0)
				original_maxDims = obj_voxels.max(axis=0)
				#print('original',numpy.sum(gt_mask), original_minDims, original_maxDims)
				mask = numpy.logical_and(numpy.all(point_voxels>=original_minDims,axis=1), numpy.all(point_voxels<=original_maxDims, axis=1))
				originalScore = 1.0 * numpy.sum(numpy.logical_and(gt_mask,mask)) / numpy.sum(numpy.logical_or(gt_mask,mask))

				#initialize the seed voxel
				seed_voxel = point_voxels[seed_id]
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
						expandPoints.extend(points[mask,:].copy())
						#determine which neighboring points should be added
						criteria = obj_id[mask] == target_id
						expandID = numpy.nonzero(mask)[0][criteria]
						expandClass.extend(criteria)
						currentMask[expandID] = True

					center = numpy.mean(currentPoints[:,:2], axis=0)
					feature_center = numpy.mean(currentPoints[:,3:], axis=0)
					currentPoints[:,:2] -= center
					stacked_points.append(currentPoints)
					stacked_count.append(len(currentPoints))
					if len(expandPoints) > 0:
						expandPoints = numpy.array(expandPoints)
						expandPoints[:,:2] -= center
						expandPoints[:,3:] -= feature_center
						stacked_neighbor_points.append(numpy.array(expandPoints))
					else:
						stacked_neighbor_points.append(numpy.zeros((0,currentPoints.shape[-1])))
					stacked_neighbor_count.append(len(expandPoints))
					stacked_class.extend(expandClass)
					steps += 1

					if numpy.sum(currentMask) == numpy.sum(gt_mask): #completed
						visited[currentMask] = True
						stacked_complete.append(1)
						stacked_steps.append(steps)
						finalScore = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
						print('AREA %d room %d target %d: %d steps %d/%d (%.2f/%.2f IOU)'%(AREA, room_id, target_id, steps, numpy.sum(currentMask), numpy.sum(gt_mask), finalScore, originalScore))
						break 
					else:
						if numpy.any(expandClass): #continue growing
							stacked_complete.append(0)
							#has matching neighbors: expand in those directions
							minDims = point_voxels[currentMask, :].min(axis=0)
							maxDims = point_voxels[currentMask, :].max(axis=0)
						else: #no matching neighbors (early termination)
							visited[currentMask] = True
							stacked_complete.append(0)
							stacked_steps.append(steps)
							finalScore = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
							print('AREA %d room %d target %d: %d steps %d/%d (%.2f/%.2f IOU)'%(AREA, room_id, target_id, steps, numpy.sum(currentMask), numpy.sum(gt_mask), finalScore, originalScore))
							break 

	if SEED is None:
		h5_fout = h5py.File('data/embedding_area%s.h5'%(AREA),'w')
	else:
		h5_fout = h5py.File('data/multiseed/embedding_seed%d_area%s.h5'%(SEED,AREA),'w')
	h5_fout.create_dataset( 'points', data=numpy.vstack(stacked_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
	h5_fout.create_dataset( 'count', data=stacked_count, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'neighbor_points', data=numpy.vstack(stacked_neighbor_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
	h5_fout.create_dataset( 'neighbor_count', data=stacked_neighbor_count, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'class', data=stacked_class, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'steps', data=stacked_steps, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'complete', data=stacked_complete, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.close()

