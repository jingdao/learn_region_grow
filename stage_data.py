from learn_region_grow_util import *
import itertools
import sys

resolution = 0.1
numpy.random.seed(0)
repeats_per_room = 1

for AREA in range(1,7):
#for AREA in [3]:
	all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
	stacked_points = []
	stacked_neighbor_points = []
	stacked_count = []
	stacked_neighbor_count = []
	stacked_class = []
	stacked_steps = []
	stacked_complete = []

#	for room_id in range(len(all_points)):
	for room_id in [0]:
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
					expandPoints = [None]*len(action_map)
					expandClass = [None]*len(action_map)
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
						expandPoints[a] = points[mask,:].copy()
						#determine which neighboring points should be added
						expandClass[a] = obj_id[mask] == target_id
						currentMask = numpy.logical_or(currentMask, numpy.logical_and(mask, obj_id==target_id))

					stacked_points.append(currentPoints)
					stacked_count.append(len(currentPoints))
					stacked_neighbor_points.extend(expandPoints)
					stacked_neighbor_count.extend([len(p) for p in expandPoints])
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
						if numpy.any(numpy.hstack(expandClass)): #continue growing
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

	normalize(stacked_points, stacked_neighbor_points)
	h5_fout = h5py.File('data/staged_area%s.h5'%AREA,'w')
	h5_fout.create_dataset( 'points', data=numpy.vstack(stacked_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
	h5_fout.create_dataset( 'count', data=stacked_count, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'neighbor_points', data=numpy.vstack(stacked_neighbor_points), compression='gzip', compression_opts=4, dtype=numpy.float32)
	h5_fout.create_dataset( 'neighbor_count', data=stacked_neighbor_count, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'class', data=numpy.hstack(stacked_class), compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'steps', data=stacked_steps, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.create_dataset( 'complete', data=stacked_complete, compression='gzip', compression_opts=4, dtype=numpy.int32)
	h5_fout.close()

