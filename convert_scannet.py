import numpy as np
import h5py

f = h5py.File('data/scannet.h5','r')
all_points = f['points'][:]
count_room = f['count_room'][:]
points = []
idp = 0
for i in range(len(count_room)):
	points.append(all_points[idp:idp+count_room[i], :])
	idp += count_room[i]
f.close()

def sample_cloud(cloud, num_samples):
    n = cloud.shape[0]
    if n >= num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
    else:
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = list(range(n)) + list(indices)
    sampled = cloud[indices, :]
    return sampled

num_points = 4096
size = 1.0
stride = 0.5
threshold = 100
for room_id in range(len(points)):
	cloud = points[room_id]
	origin = np.amin(cloud, axis=0)[0:3]
	cloud[:, 0:3] -= origin
	cloud[:, 3:6] = cloud[:, 3:6] + 0.5
	cloud[:, 7] = cloud[:, 6]
	cloud[:, 6] = 0

	limit = np.amax(cloud[:, 0:3], axis=0)
	width = int(np.ceil((limit[0] - size) / stride)) + 1
	depth = int(np.ceil((limit[1] - size) / stride)) + 1
	cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
	blocks = []
	for (x, y) in cells:
		xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
		ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
		cond  = xcond & ycond
		if np.sum(cond) < threshold:
			continue
		block = cloud[cond, :]
		block = sample_cloud(block, num_points)
		blocks.append(block)
	blocks = np.stack(blocks, axis=0)
	# A batch should have shape of BxNx14, where
	# [0:3] - global coordinates
	# [3:6] - block normalized coordinates (centered at Z-axis)
	# [6:9] - RGB colors
	# [9:12] - room normalized coordinates
	# [12:14] - semantic and instance labels
	num_blocks = blocks.shape[0]
	batch = np.zeros((num_blocks, num_points, 14))
	for b in range(num_blocks):
		minx = min(blocks[b, :, 0])
		miny = min(blocks[b, :, 1])
		batch[b, :, 3]  = blocks[b, :, 0] - (minx + size * 0.5)
		batch[b, :, 4]  = blocks[b, :, 1] - (miny + size * 0.5)
		batch[b, :, 9]  = blocks[b, :, 0] / limit[0]
		batch[b, :, 10] = blocks[b, :, 1] / limit[1]
		batch[b, :, 11] = blocks[b, :, 2] / limit[2]
	batch[:,:, 0:3] = blocks[:,:,0:3]
	batch[:,:, 5:9] = blocks[:,:,2:6]
	batch[:,:, 12:] = blocks[:,:,6:8]
	print(room_id, cloud.shape, batch.shape)

	fname = '/home/jd/Downloads/jsis3d/data/s3dis/h5/ScanNet_room_%d.h5' % room_id
	fp = h5py.File(fname, 'w')
	coords = batch[:, :, 0:3]
	pts = batch[:, :, 3:12]
	labels = batch[:, :, 12:14]
	fp.create_dataset('coords', data=coords, compression='gzip', dtype='float32')
	fp.create_dataset('points', data=pts, compression='gzip', dtype='float32')
	fp.create_dataset('labels', data=labels, compression='gzip', dtype='int64')
	fp.close()
