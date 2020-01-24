from learn_region_grow_util import loadFromH5, savePCD
import h5py
import itertools
import sys
import numpy as np
from class_util import classes

resolution = 0.1
SEED = None
cluster_threshold = 10
max_points = 1024
save_id = 0
np.random.seed(0)
for i in range(len(sys.argv)):
	if sys.argv[i]=='--seed':
		SEED = int(sys.argv[i+1])
		np.random.seed(SEED)

for AREA in range(1,7):
#for AREA in [1]:
	all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
	stacked_points = []
	stacked_neighbor_points = []
	stacked_count = []
	stacked_neighbor_count = []
	stacked_remove = []
	stacked_add = []
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
		xyz = points[:,:3]
		rgb = points[:,3:6]
		room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

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
		curvatures = curvatures/curvatures.max()
		points = np.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(np.float32)

		point_voxels = np.round(points[:,:3]/resolution).astype(int)
		visited = np.zeros(len(point_voxels), dtype=bool)
		#iterate over each voxel in the room
		for seed_id in np.random.choice(range(len(points)), len(points), replace=False):
#		for seed_id in np.arange(len(points))[np.argsort(curvatures)]:
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
			stuck = False
			add_mistake_prob = np.random.randint(2,5)*0.1
			remove_mistake_prob = np.random.randint(2,5)*0.1

			#perform region growing
			while True:

				#determine the current points and the neighboring points
				currentPoints = points[currentMask, :].copy()
				newMinDims = minDims.copy()	
				newMaxDims = maxDims.copy()	
				newMinDims -= 1
				newMaxDims += 1
				mask = np.logical_and(np.all(point_voxels>=newMinDims,axis=1), np.all(point_voxels<=newMaxDims, axis=1))
				mask = np.logical_and(mask, np.logical_not(currentMask))
				mask = np.logical_and(mask, np.logical_not(visited))

				#determine which points to accept
				expandPoints = points[mask, :].copy()
				expandClass = obj_id[mask] == target_id
				mask_idx = np.nonzero(mask)[0]
				if stuck:
					expandID = mask_idx[expandClass]
				else:
					mistake_sample = np.random.random(len(mask_idx)) < add_mistake_prob
					expand_with_mistake = np.logical_xor(expandClass, mistake_sample)
					expandID = mask_idx[expand_with_mistake]

				#determine which points to reject
				rejectClass = obj_id[currentMask] != target_id
				mask_idx = np.nonzero(currentMask)[0]
				if stuck:
					rejectID = mask_idx[rejectClass]
				else:
					mistake_sample = np.random.random(len(mask_idx)) < remove_mistake_prob
					reject_with_mistake = np.logical_xor(rejectClass, mistake_sample)
					rejectID = mask_idx[reject_with_mistake]

				#update current mask
				currentMask[expandID] = True
				if len(rejectID) < len(mask_idx):
					currentMask[rejectID] = False
				nextMinDims = point_voxels[currentMask, :].min(axis=0)
				nextMaxDims = point_voxels[currentMask, :].max(axis=0)
				if not np.any(nextMinDims<minDims) and not np.any(nextMaxDims>maxDims):
					stuck = True
				iou = 1.0 * np.sum(np.logical_and(currentMask, gt_mask)) / np.sum(np.logical_or(currentMask, gt_mask))
				print('mask %d/%d/%d expand %d/%d reject %d/%d iou %.2f'%(np.sum(np.logical_and(currentMask, gt_mask)),
					np.sum(currentMask), np.sum(gt_mask), len(expandID), np.sum(expandClass), len(rejectID), np.sum(rejectClass), iou))

				if len(expandPoints) > 0:
					if len(currentPoints) <= max_points:
						stacked_points.append(currentPoints)
						stacked_count.append(len(currentPoints))
						stacked_remove.extend(rejectClass)
					else:
						subset = np.random.choice(len(currentPoints), max_points, replace=False)
						stacked_points.append(currentPoints[subset])
						stacked_count.append(max_points)
						stacked_remove.extend(rejectClass[subset])
					if len(expandPoints) <= max_points:
						stacked_neighbor_points.append(np.array(expandPoints))
						stacked_neighbor_count.append(len(expandPoints))
						stacked_add.extend(expandClass)
					else:
						subset = np.random.choice(len(expandPoints), max_points, replace=False)
						stacked_neighbor_points.append(expandPoints[subset])
						stacked_neighbor_count.append(max_points)
						stacked_add.extend(expandClass[subset])
					stacked_complete.append(iou)
					steps += 1
					add_mistake_prob = max(add_mistake_prob-0.01, 0.00)
					remove_mistake_prob = max(remove_mistake_prob-0.01, 0.00)

				if np.all(currentMask == gt_mask): #completed
					visited[currentMask] = True
					stacked_steps.append(steps)
#					savePCD('tmp/%d-cloud.pcd'%save_id, points[currentMask])
#					save_id += 1
					print('AREA %d room %d target %d: %d steps %d/%d (%.2f/%.2f IOU)'%(AREA, room_id, target_id, steps, np.sum(currentMask), np.sum(gt_mask), iou, originalScore))
					break 
				else:
					if np.any(expandClass) or np.any(rejectClass): #continue growing
						#has matching neighbors: expand in those directions
						minDims = nextMinDims
						maxDims = nextMaxDims
					else: #no matching neighbors (early termination)
						if np.sum(currentMask) > cluster_threshold:
							visited[currentMask] = True
							stacked_steps.append(steps)
#							savePCD('tmp/%d-cloud.pcd'%save_id, points[currentMask])
#							save_id += 1
							print('AREA %d room %d target %d: %d steps %d/%d (%.2f/%.2f IOU)'%(AREA, room_id, target_id, steps, np.sum(currentMask), np.sum(gt_mask), iou, originalScore))
						break 

	for i in range(len(stacked_points)):
		center = np.mean(stacked_points[i][:,:2], axis=0)
		stacked_points[i][:,:2] -= center
		feature_center = np.mean(stacked_points[i][:,6:], axis=0)
		if len(stacked_neighbor_points[i]) > 0:
			stacked_neighbor_points[i][:,:2] -= center
			stacked_neighbor_points[i][:,6:] -= feature_center

	if SEED is None:
		h5_fout = h5py.File('data/staged_area%s.h5'%(AREA),'w')
#		h5_fout = h5py.File('data/small_area%s.h5'%(AREA),'w')
	else:
		h5_fout = h5py.File('data/multiseed/seed%d_area%s.h5'%(SEED,AREA),'w')
	h5_fout.create_dataset( 'points', data=np.vstack(stacked_points), compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.create_dataset( 'count', data=stacked_count, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'neighbor_points', data=np.vstack(stacked_neighbor_points), compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.create_dataset( 'neighbor_count', data=stacked_neighbor_count, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'add', data=stacked_add, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'remove', data=stacked_remove, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'steps', data=stacked_steps, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'complete', data=stacked_complete, compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.close()

