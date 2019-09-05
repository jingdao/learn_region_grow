import h5py
import sys
import numpy

def loadFromH5(filename):
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
	obj_labels = []
	class_labels = []
	for i in range(len(tmp_points)):
		room.append(tmp_points[i][:,:-2])
		obj_labels.append(tmp_points[i][:,-2].astype(int))
		class_labels.append(tmp_points[i][:,-1].astype(int))
	return room, obj_labels, class_labels

def savePLY(filename, points):
	f = open(filename,'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar r
property uchar g
property uchar b
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

numRooms = 0
combined_points = []
all_room_id = None

mode = 'rgb'
if '--rgb' in sys.argv:
	mode = 'rgb'
if '--seg' in sys.argv:
	mode = 'seg'

all_points, all_obj_id, all_cls_id = loadFromH5(sys.argv[1])

for room_id in range(len(all_points)):
	points = all_points[room_id]
	obj_id = all_obj_id[room_id]
	cls_id = all_cls_id[room_id]
	if mode=='rgb':
		points[:,3:6] = (points[:,3:6]+0.5)*255
		savePLY('data/viz/%d.ply'%room_id, points)
	elif mode=='seg':
		obj_color = numpy.random.randint(0,255,(numpy.max(obj_id)+1,3))
		obj_color[0,:] = [200,200,200]
		points[:,3:6] = obj_color[obj_id,:]
		savePLY('data/viz/%d.ply'%room_id, points)
