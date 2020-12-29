import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from class_util import classes, class_to_id, class_to_color_rgb
import itertools
import math
import networkx as nx
import time
import matplotlib.pyplot as plt
import scipy.special
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont
from learn_region_grow_util import LrgNet

numpy.random.seed(0)
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
FEATURE_SIZE = 13
TEST_AREAS = [1,2,3,4,5,6,'scannet']
resolution = 0.1
completion_threshold = 0.5
classification_threshold = 0.5
cluster_threshold = 10
AREA = 5
ROOM = 44
color_sample_state = numpy.random.RandomState(0)
instance_color_id = color_sample_state.randint(0,255,(20,3))

for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		AREA = int(sys.argv[i+1])
	if sys.argv[i]=='--room':
		ROOM = int(sys.argv[i+1])

cameraX = 9.42495995258856
cameraY = 9.381724865127635
cameraZ = 4.651620026309769
centerX=0
centerY=0
centerZ=0
upX=0
upY=0
upZ=1
mouseIndex = 0
previousX = 0
previousY = 0
scrollSpeed = 1.1
fov = 70
img_id = 0

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

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
	f.close()
	print('Saved %d points to %s' % (l,filename))


all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%s.h5' % AREA)
room_id = ROOM

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
points = unequalized_points[equalized_idx]
obj_id = obj_id[equalized_idx]
cls_id = cls_id[equalized_idx]
xyz = points[:,:3]
rgb = points[:,3:6]
room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

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
	curvatures.append(numpy.fabs(curvature))
curvatures = numpy.array(curvatures)
curvatures = curvatures/curvatures.max()
normals = numpy.array(normals)
points = numpy.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(numpy.float32)
result_points = unequalized_points.copy()
result_points[:,3:6] = (result_points[:,3:6]+0.5)*255
viz_points = []
instance_color = numpy.ones((len(points), 3), dtype=int)*100
print('points',points.shape)

def displayFun():
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	gluLookAt(cameraX,cameraY,cameraZ,centerX,centerY,centerZ,upX,upY,upZ)

	glPointSize(5.0)
	glBegin(GL_POINTS)
	for n in range(len(viz_points)):
		glColor3ub(int(viz_points[n][3]),int(viz_points[n][4]),int(viz_points[n][5]))
		glVertex3d(viz_points[n][0],viz_points[n][1],viz_points[n][2])
	for n in range(len(result_points)):
		glColor3ub(int(result_points[n][3]),int(result_points[n][4]),int(result_points[n][5]))
		glVertex3d(result_points[n][0],result_points[n][1],result_points[n][2])
	glEnd()

	glFlush()
	glutSwapBuffers()

#takes mouse click/wheel input to change the view angle
def mouseFun(button,state,x,y):
	global cameraX,cameraY,cameraZ,previousX,previousY
	if button==3:
		cameraX /= scrollSpeed
		cameraY /= scrollSpeed
		cameraZ /= scrollSpeed
	elif button==4:
		cameraX *= scrollSpeed
		cameraY *= scrollSpeed
		cameraZ *= scrollSpeed
	elif button==GLUT_LEFT_BUTTON and state == GLUT_DOWN:
		mouseIndex = button
		previousX = x
		previousY = y
	glutPostRedisplay()

#takes mouse drag input to change the view angle
def motionFun(x,y):
	global cameraX,cameraY,cameraZ,previousX,previousY
	if mouseIndex == GLUT_LEFT_BUTTON:
		rho = math.sqrt(cameraX*cameraX+cameraY*cameraY)
		xstep = cameraY / rho
		ystep = -cameraX / rho
		cameraX += 0.05 * (x-previousX) * xstep
		cameraY += 0.05 * (x-previousX) * ystep
		cameraZ += 0.05 * (y-previousY)
		previousX = x
		previousY = y
		glutPostRedisplay()

def keyFun(key,x,y):
	global cameraX,cameraY,cameraZ
	if key==b' ':
		cameraX=5
		cameraY=5
		cameraZ=5
		glutPostRedisplay()
	elif key==b'w':
		cameraX /= scrollSpeed
		cameraY /= scrollSpeed
		cameraZ /= scrollSpeed
		glutPostRedisplay()
	elif key=='s':
		cameraX *= scrollSpeed
		cameraY *= scrollSpeed
		cameraZ *= scrollSpeed
		glutPostRedisplay()
	elif key=='c':
		print(cameraX,cameraY,cameraZ)

def reshapeFun(w, h):
    glViewport (0, 0, w, h)
	
width = height = 1000
glutInit()
glutInitWindowSize(width, height)
glutCreateWindow("ANIMATE")
#glutHideWindow()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
glutDisplayFunc(displayFun)
glutReshapeFunc(reshapeFun)
glutKeyboardFunc(keyFun)
glutMouseFunc(mouseFun)
glutMotionFunc(motionFun)
glutIdleFunc(None)
glEnable(GL_DEPTH_TEST)

glClearColor(0.0,0.0,0.0,0.0)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(fov,600.0/600,1,1000)
#glutMainLoop()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = LrgNet(1, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
saver = tf.train.Saver()
MODEL_PATH = 'models/lrgnet_model%s.ckpt'%AREA
saver.restore(sess, MODEL_PATH)
print('Restored network from %s'%MODEL_PATH)

result_points[:,3:6] = [100,100,100]
point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
cluster_label = numpy.zeros(len(points), dtype=int)
cluster_id = 1
visited = numpy.zeros(len(point_voxels), dtype=bool)
inlier_points = numpy.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=numpy.float32)
neighbor_points = numpy.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=numpy.float32)
input_add = numpy.zeros((1, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
input_remove = numpy.zeros((1, NUM_INLIER_POINT), dtype=numpy.int32)
#iterate over each object in the room
#for seed_id in range(len(point_voxels)):
for seed_id in numpy.arange(len(points))[numpy.argsort(curvatures)]:
	if visited[seed_id]:
		continue
	seed_voxel = point_voxels[seed_id]
	target_id = obj_id[seed_id]
	target_class = classes[cls_id[numpy.nonzero(obj_id==target_id)[0][0]]]
	gt_mask = obj_id==target_id
	obj_voxels = point_voxels[gt_mask]
	obj_voxel_set = set([tuple(p) for p in obj_voxels])
	original_minDims = obj_voxels.min(axis=0)
	original_maxDims = obj_voxels.max(axis=0)
	currentMask = numpy.zeros(len(points), dtype=bool)
	currentMask[seed_id] = True
	minDims = seed_voxel.copy()
	maxDims = seed_voxel.copy()
	seqMinDims = minDims
	seqMaxDims = maxDims
	steps = 0
	stuck = 0

	#perform region growing
	while True:

		def stop_growing(reason):
			global cluster_id
			visited[currentMask] = True
			if numpy.sum(currentMask) > cluster_threshold:
				cluster_label[currentMask] = cluster_id
				cluster_id += 1
				iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
				print('room %d target %3d %.4s: step %3d %4d/%4d points IOU %.3f add %.3f rmv %.3f %s'%(room_id, target_id, target_class, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, add_acc, rmv_acc, reason))

		#determine the current points and the neighboring points
		currentPoints = points[currentMask, :].copy()
		newMinDims = minDims.copy()	
		newMaxDims = maxDims.copy()	
		newMinDims -= 1
		newMaxDims += 1
		mask = numpy.logical_and(numpy.all(point_voxels>=newMinDims,axis=1), numpy.all(point_voxels<=newMaxDims, axis=1))
		mask = numpy.logical_and(mask, numpy.logical_not(currentMask))
		mask = numpy.logical_and(mask, numpy.logical_not(visited))
		expandPoints = points[mask, :].copy()
		expandClass = obj_id[mask] == target_id
		rejectClass = obj_id[currentMask] != target_id
		
		if len(expandPoints)==0: #no neighbors (early termination)
			stop_growing('noneighbor')
			break 

		if len(currentPoints) >= NUM_INLIER_POINT:
			subset = numpy.random.choice(len(currentPoints), NUM_INLIER_POINT, replace=False)
		else:
			subset = list(range(len(currentPoints))) + list(numpy.random.choice(len(currentPoints), NUM_INLIER_POINT-len(currentPoints), replace=True))
		center = numpy.median(currentPoints, axis=0)
		expandPoints = numpy.array(expandPoints)
		expandPoints[:,:2] -= center[:2]
		expandPoints[:,6:] -= center[6:]
		inlier_points[0,:,:] = currentPoints[subset, :]
		inlier_points[0,:,:2] -= center[:2]
		inlier_points[0,:,6:] -= center[6:]
		input_remove[0,:] = numpy.array(rejectClass)[subset]
		if len(expandPoints) >= NUM_NEIGHBOR_POINT:
			subset = numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
		else:
			subset = list(range(len(expandPoints))) + list(numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
		neighbor_points[0,:,:] = numpy.array(expandPoints)[subset, :]
		input_add[0,:] = numpy.array(expandClass)[subset]
		ls, add,add_acc, rmv,rmv_acc = sess.run([net.loss, net.add_output, net.add_acc, net.remove_output, net.remove_acc],
			{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})

		add_conf = scipy.special.softmax(add[0], axis=-1)[:,1]
		rmv_conf = scipy.special.softmax(rmv[0], axis=-1)[:,1]
		add_mask = numpy.random.random(len(add_conf)) < add_conf
		rmv_mask = numpy.random.random(len(rmv_conf)) < rmv_conf
		addPoints = neighbor_points[0,:,:][add_mask]
		addPoints[:,:2] += center[:2]
		addVoxels = numpy.round(addPoints[:,:3]/resolution).astype(int)
		addSet = set([tuple(p) for p in addVoxels])
		rmvPoints = inlier_points[0,:,:][rmv_mask]
		rmvPoints[:,:2] += center[:2]
		rmvVoxels = numpy.round(rmvPoints[:,:3]/resolution).astype(int)
		rmvSet = set([tuple(p) for p in rmvVoxels])
		updated = False
		for i in range(len(point_voxels)):
			if not currentMask[i] and tuple(point_voxels[i]) in addSet:
				currentMask[i] = True
				updated = True
			if tuple(point_voxels[i]) in rmvSet:
				currentMask[i] = False
		steps += 1
		
		#slightly rotate camera
#		rho = math.sqrt(cameraX*cameraX+cameraY*cameraY)
#		theta = numpy.arctan2(cameraY, cameraX)
#		theta += 0.01
#		cameraX = rho * numpy.cos(theta)
#		cameraY = rho * numpy.sin(theta)
		obj_color = numpy.ones((len(points), 3)) * 100
		obj_color[currentMask] = [0, 255, 0]
#		obj_color[currentMask] = [150, 150, 150]
		obj_color[mask] = [0, 0, 255]
		jet = plt.get_cmap('jet')
		result_points[:,3:6] = obj_color[unequalized_idx]
#		viz_points = neighbor_points[0,:,:]
#		viz_points[:,:2] += center[:2]
#		viz_points[:,3:6] = jet(cls_conf)[:,:3] * 255
		displayFun()
		data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
		image = Image.frombytes("RGBA", (width, height), data)
		image = ImageOps.flip(image)
		d = ImageDraw.Draw(image)
		fnt = ImageFont.truetype('FreeMono.ttf', 40)
		d.text((10,10), "Step %d"%steps, font=fnt, fill=(255,255,255,255))
		d.text((10,60), "Inlier Set: %d points"%(numpy.sum(currentMask)), font=fnt, fill=(255,255,255,255))
		d.text((10,110), "Neighbor Set: %d points"%(numpy.sum(mask)), font=fnt, fill=(255,255,255,255))
		d.text((10,160), "Add: %d points"%len(addSet), font=fnt, fill=(255,255,255,255))
		d.text((10,210), "Remove: %d points"%len(rmvSet), font=fnt, fill=(255,255,255,255))
		image.save('tmp/step%03d.png' % img_id, 'PNG')
		instance_color[currentMask] = instance_color_id[cluster_id]
		result_points[:,3:6] = instance_color[unequalized_idx]
		displayFun()
		data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
		image = Image.frombytes("RGBA", (width, height), data)
		image = ImageOps.flip(image)
		image.save('tmp/seg%03d.png' % img_id, 'PNG')
#		print('Saved image %d'%img_id)
		img_id += 1

		if updated: #continue growing
			minDims = point_voxels[currentMask, :].min(axis=0)
			maxDims = point_voxels[currentMask, :].max(axis=0)
			if not numpy.any(minDims<seqMinDims) and not numpy.any(maxDims>seqMaxDims):
				if stuck >= 1:
					stop_growing('stuck')
					break
				else:
					stuck += 1
			else:
				stuck = 0
			seqMinDims = numpy.minimum(seqMinDims, minDims)
			seqMaxDims = numpy.maximum(seqMaxDims, maxDims)
		else: #no matching neighbors (early termination)
			stop_growing('noexpand')
			break

#	break

