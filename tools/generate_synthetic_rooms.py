import numpy
import h5py
from learn_region_grow_util import loadFromH5, savePCD
import sys

if False:
	room_dimensions = []
	color_variation = []

	for AREA in range(1,7):
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
		for room_id in range(len(all_points)):
			unequalized_points = all_points[room_id]
			obj_id = all_obj_id[room_id]
			cls_id = all_cls_id[room_id]
			room_dimensions.append(unequalized_points[:,:3].max(axis=0) - unequalized_points[:,:3].min(axis=0))
			for i in set(obj_id):
				obj_points = unequalized_points[obj_id==i]
				color_variation.append(numpy.std(obj_points[:,3:6], axis=0))

	room_dimensions = numpy.array(room_dimensions)
	color_variation = numpy.array(color_variation)
	room_variation = numpy.std(room_dimensions, axis=0)
	room_min = numpy.min(room_dimensions, axis=0)
	room_max = numpy.max(room_dimensions, axis=0)
	room_dimensions = numpy.mean(room_dimensions, axis=0)
	color_variation = numpy.mean(color_variation, axis=0)
	print('room_dimensions', room_dimensions)
	print('room_variation', room_variation)
	print('room_min', room_min)
	print('room_max', room_max)
	print('color_variation', color_variation)
	sys.exit(0)

room_min = numpy.array([1.0619999, 1.0630007, 2.073])
room_max = numpy.array([44.094, 46.835,  7.647])
room_dimensions = numpy.array([5.133024 , 5.169554 , 3.0433161])
room_variation = numpy.array([4.2353425, 5.5636344, 0.58006])
color_variation = numpy.array([0.15274304, 0.15051211, 0.15046296])

def generate_room(width, length, height):
	density = 0.05
	xyz_noise = 0.01
	room = []
	def applyNoiseAndColor(P):
		P[:,:3] += numpy.random.randn(len(P), 3) * xyz_noise
		mean_color = numpy.random.random(3) - 0.5
		P[:,3:6] = mean_color + numpy.random.randn(len(P), 3) * color_variation * 0.5
		P[:,3:6] = numpy.minimum(0.5, P[:,3:6])
		P[:,3:6] = numpy.maximum(-0.5, P[:,3:6])
	#floor
	N = int(width * length / density**2)
	pcd = numpy.zeros((N, 8))
	pcd[:,0] = numpy.random.random(N)*width
	pcd[:,1] = numpy.random.random(N)*length
	pcd[:,6] = 1
	applyNoiseAndColor(pcd)
	room.extend(pcd)
	#ceiling
	pcd = numpy.zeros((N, 8))
	pcd[:,0] = numpy.random.random(N)*width
	pcd[:,1] = numpy.random.random(N)*length
	pcd[:,2] = height
	pcd[:,6] = 2
	applyNoiseAndColor(pcd)
	room.extend(pcd)
	#back wall
	N = int(width * height / density**2)
	pcd = numpy.zeros((N, 8))
	pcd[:,0] = numpy.random.random(N)*width
	pcd[:,2] = numpy.random.random(N)*height
	pcd[:,6] = 3
	applyNoiseAndColor(pcd)
	room.extend(pcd)
	#front wall
	pcd = numpy.zeros((N, 8))
	pcd[:,0] = numpy.random.random(N)*width
	pcd[:,1] = length
	pcd[:,2] = numpy.random.random(N)*height
	pcd[:,6] = 4
	applyNoiseAndColor(pcd)
	room.extend(pcd)
	#left wall
	N = int(length * height / density**2)
	pcd = numpy.zeros((N, 8))
	pcd[:,1] = numpy.random.random(N)*length
	pcd[:,2] = numpy.random.random(N)*height
	pcd[:,6] = 5
	applyNoiseAndColor(pcd)
	room.extend(pcd)
	#right wall
	pcd = numpy.zeros((N, 8))
	pcd[:,0] = width
	pcd[:,1] = numpy.random.random(N)*length
	pcd[:,2] = numpy.random.random(N)*height
	pcd[:,6] = 6
	applyNoiseAndColor(pcd)
	room.extend(pcd)
	return numpy.array(room)

area = []
for room_id in range(20):
	print('train',room_id)
	wlh = room_dimensions + numpy.random.randn(3)*room_variation
	wlh = numpy.maximum(room_min, wlh)
	wlh = numpy.minimum(room_max, wlh)
	room = generate_room(wlh[0], wlh[1], wlh[2])
	area.append(room)
#	room[:,3:6] = (room[:,3:6]+0.5)*255
#	savePCD('tmp/%d-cloud.pcd'%room_id, room)
count = [len(p) for p in area]
h5_fout = h5py.File('data/synthetic_train.h5','w')
h5_fout.create_dataset( 'points', data=numpy.vstack(area), compression='gzip', compression_opts=4, dtype=numpy.float32)
h5_fout.create_dataset( 'count_room', data=count, compression='gzip', compression_opts=4, dtype=numpy.int32)
h5_fout.close()
area = []
for room_id in range(5):
	print('test',room_id)
	wlh = room_dimensions + numpy.random.randn(3)*room_variation
	wlh = numpy.maximum(room_min, wlh)
	wlh = numpy.minimum(room_max, wlh)
	room = generate_room(wlh[0], wlh[1], wlh[2])
	area.append(room)
count = [len(p) for p in area]
h5_fout = h5py.File('data/synthetic_test.h5','w')
h5_fout.create_dataset( 'points', data=numpy.vstack(area), compression='gzip', compression_opts=4, dtype=numpy.float32)
h5_fout.create_dataset( 'count_room', data=count, compression='gzip', compression_opts=4, dtype=numpy.int32)
h5_fout.close()
