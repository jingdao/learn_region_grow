import numpy
import h5py

def triangleArea(p1,p2,p3):
	v1=p2-p1
	v2=p3-p1
	area=0.5*numpy.linalg.norm(numpy.cross(v1,v2))
	return area

def uniform_sample():
	if obj_name.startswith('DataDevice') or obj_name.startswith('Light-Surface') or 'Surface' in obj_name:
		return
	print('%d: Processing %d vertices %d faces from %s'%(numObjects,len(vertices),len(faces),obj_name[:20]))
	for f in faces:
		p1 = vertices[f[0]]
		p2 = vertices[f[1]]
		p3 = vertices[f[2]]
		v1=p2-p1
		v2=p3-p1
		v3=v1+v2
		area=triangleArea(p1,p2,p3)
		numSamples = area/density
		r = numSamples - int(numSamples)
		numSamples = int(numSamples)
		if numpy.random.random() < r:
			numSamples += 1
		for n in range(numSamples):
			a=numpy.random.random()
			b=numpy.random.random()
			x = p1 + a*v1 + b*v2
			A1 = triangleArea(p1,p2,x)
			A2 = triangleArea(p1,p3,x)
			A3 = triangleArea(p2,p3,x)
			if abs(A1 + A2 + A3 - area) > 1e-6:
				x = p1 + v3 - a*v1 - b*v2
			points.append(x)
			labels.append(numObjects)

density=0.03
buildingID = 0
count_room = []
stacked_points = None
for filename in [
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/JE DUNN office file/JEDunn_sample.obj',
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/JE DUNN office file/advanced_sample.obj',
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/JE DUNN office file/sample.obj',
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/BeckGroup Data/02_BIMModels/BeckRevitModelsIFC/crescent.obj',
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/BeckGroup Data/02_BIMModels/BeckRevitModelsIFC/unt.obj',
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/BeckGroup Data/02_BIMModels/BeckRevitModelsIFC/georgia.obj',
		'/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/AjaxData/BR-30-1607_GTCSF_ARCH_CENTRAL_R16-3DView-{3D-jchen490}.obj',
	]:
	vertices=[]
	faces=[]
	points=[]
	labels=[]
	numObjects=0
	vOffset=1
	scale = 0.3048 if "Ajax" in filename or "sample" in filename else 1
	print('Opening',filename)
	f = open(filename,'r')
	for l in f:
		if l.startswith('g '):
			if len(faces) > 0:
				uniform_sample()
				numObjects += 1
				vOffset += len(vertices)
				vertices = []
				faces = []
			obj_name = l.split()[1]
		elif l.startswith('v '):
			vertices.append(numpy.array([float(t) for t in l.split()[1:]]) * scale)
		elif l.startswith('f '):
			faces.append([int(t.split('/')[0])-vOffset for t in l.split()[1:]])
	f.close()
	if len(faces) > 0:
		uniform_sample()

	if len(points)==0:
		continue
	cloud = numpy.zeros((len(points),8))
	cloud[:,:3] = points
	cloud[:,6] = labels
	centroid = cloud[:,:2].mean(axis=0)
	cloud[:,:2] -= centroid

#	obj_color = numpy.random.randint(0,255,(numpy.max(labels)+1,3))
#	cloud[:,3:6] = obj_color[labels, :]
#	numpy.save('data/construction_embedding/building%d.npy'%buildingID,cloud)
#	savePCD('data/construction_embedding/%d-cloud.pcd'%buildingID,cloud)
	buildingID += 1
	if stacked_points is None:
		stacked_points = cloud
	else:
		stacked_points = numpy.vstack((stacked_points, cloud))
	count_room.append(len(cloud))

h5_fout = h5py.File('data/bim.h5','w')
h5_fout.create_dataset( 'points', data=stacked_points, compression='gzip', compression_opts=4, dtype=numpy.float32)
h5_fout.create_dataset( 'count_room', data=numpy.array(count_room), compression='gzip', compression_opts=4, dtype=numpy.int32)
h5_fout.close()
