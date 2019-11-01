from learn_region_grow_util import *
import itertools
import sys
import numpy as np
from class_util import classes

resolution = 0.1
np.random.seed(0)
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
			k = tuple(np.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_set:
				equalized_set.add(k)
				equalized_idx.append(i)
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
		# points -> XYZ + RGB
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]

		#compute normals and curvatures
		normals = []
		curvatures = []
		for i in range(len(points)):
			k = tuple(np.round(points[i,:3]/resolution).astype(int))
			neighbors = []
			for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
				kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
				if kk in normal_grid:
					neighbors.extend(normal_grid[kk])
			accA = np.zeros((3,3))
			accB = np.zeros(3)
			for n in neighbors:
				p = unequalized_points[n,:3]
				accA += np.outer(p,p)
				accB += p
			cov = accA / len(neighbors) - np.outer(accB, accB) / len(neighbors)**2
			U,S,V = np.linalg.svd(cov)
			normals.append(np.fabs(V[2]))
			curvature = S[2] / (S[0] + S[1] + S[2])
			curvatures.append(np.fabs(curvature)) 
		normals = np.array(normals)
		curvatures = np.array(curvatures)
		## XYZ
		# points = points[:,0:3]

		## XYZ + RGB + normal(x,y,z)
#		points = np.hstack((points, normals)).astype(np.float32)
		
		## XYZ + RGB + normal(x,y,z) + curvature
		points = np.hstack((points, normals, curvatures.reshape(-1,1))).astype(np.float32)

		point_voxels = np.round(points[:,:3]/resolution).astype(int)
		for i in range(repeats_per_room):
			visited = np.zeros(len(point_voxels), dtype=bool)
			#iterate over each voxel in the room
			#for seed_id in np.random.choice(range(len(points)), len(points), replace=False):
			for seed_id in np.arange(len(points))[np.argsort(curvatures)]:
				if visited[seed_id]:
					continue
				target_id = obj_id[seed_id]
				obj_voxels = point_voxels[obj_id==target_id, :]
				gt_mask = obj_id==target_id
				original_minDims = obj_voxels.min(axis=0)
				original_maxDims = obj_voxels.max(axis=0)
				#print('original',np.sum(gt_mask), original_minDims, original_maxDims)
				mask = np.logical_and(np.all(point_voxels>=original_minDims,axis=1), np.all(point_voxels<=original_maxDims, axis=1))
				originalScore = 1.0 * np.sum(np.logical_and(gt_mask,mask)) / np.sum(np.logical_or(gt_mask,mask))

				#initialize the seed voxel
				seed_voxel = point_voxels[seed_id]
				currentMask = np.zeros(len(points), dtype=bool)
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
							mask = np.logical_and(np.all(point_voxels>=minDims,axis=1), np.all(point_voxels<=maxDims, axis=1))
							mask = np.logical_and(mask, np.logical_not(currentMask))
						else:
							newMinDims = minDims.copy()	
							newMaxDims = maxDims.copy()	
							expand_dim = np.nonzero(action_map[a])[0][0] % 3
							if np.sum(action_map[a])>0:
								newMinDims[expand_dim] = newMaxDims[expand_dim] = maxDims[expand_dim]+1
							else:
								newMinDims[expand_dim] = newMaxDims[expand_dim] = minDims[expand_dim]-1
							mask = np.logical_and(np.all(point_voxels>=newMinDims,axis=1), np.all(point_voxels<=newMaxDims, axis=1))
						expandPoints.extend(points[mask,:].copy())
						#determine which neighboring points should be added
						criteria = obj_id[mask] == target_id
#						avg_normal = currentPoints[:,6:9].mean(axis=0)
#						criteria = normals[mask].dot(avg_normal) > 0.99
						expandID = np.nonzero(mask)[0][criteria]
						expandClass.extend(criteria)
						currentMask[expandID] = True

					stacked_points.append(currentPoints)
					stacked_count.append(len(currentPoints))
					if len(expandPoints) > 0:
						stacked_neighbor_points.append(np.array(expandPoints))
					else:
						stacked_neighbor_points.append(np.zeros((0,currentPoints.shape[-1])))
					stacked_neighbor_count.append(len(expandPoints))
					stacked_class.extend(expandClass)
					steps += 1

					if np.sum(currentMask) == np.sum(gt_mask): #completed
						visited[currentMask] = True
						stacked_complete.append(1)
						stacked_steps.append(steps)
						finalScore = 1.0 * np.sum(np.logical_and(gt_mask,currentMask)) / np.sum(np.logical_or(gt_mask,currentMask))
						print('AREA %d room %d target %d: %d steps %d/%d (%.2f/%.2f IOU)'%(AREA, room_id, target_id, steps, np.sum(currentMask), np.sum(gt_mask), finalScore, originalScore))
						break 
					else:
						if np.any(expandClass): #continue growing
							stacked_complete.append(0)
							#has matching neighbors: expand in those directions
							minDims = point_voxels[currentMask, :].min(axis=0)
							maxDims = point_voxels[currentMask, :].max(axis=0)
						else: #no matching neighbors (early termination)
							visited[currentMask] = True
							stacked_complete.append(0)
							stacked_steps.append(steps)
							finalScore = 1.0 * np.sum(np.logical_and(gt_mask,currentMask)) / np.sum(np.logical_or(gt_mask,currentMask))
							print('AREA %d room %d target %d: %d steps %d/%d (%.2f/%.2f IOU)'%(AREA, room_id, target_id, steps, np.sum(currentMask), np.sum(gt_mask), finalScore, originalScore))
							break 

	normalize(stacked_points, stacked_neighbor_points)
	h5_fout = h5py.File('data/staged_area%s.h5'%AREA,'w')
	h5_fout.create_dataset( 'points', data=np.vstack(stacked_points), compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.create_dataset( 'count', data=stacked_count, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'neighbor_points', data=np.vstack(stacked_neighbor_points), compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.create_dataset( 'neighbor_count', data=stacked_neighbor_count, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'class', data=stacked_class, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'steps', data=stacked_steps, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'complete', data=stacked_complete, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.close()

