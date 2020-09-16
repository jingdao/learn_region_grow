import numpy

#list of all classes
classes = ['clutter', 'board', 'bookcase', 'beam', 'chair', 'column', 'door', 'sofa', 'table', 'window', 'ceiling', 'floor', 'wall']

#integer ID for each class
class_to_id = {classes[i] : i for i in range(len(classes))}

#minimum percentage of object points that has to be in a cell 
point_ratio_threshold = {
	'clutter': 0,
	'board': 0.1,
	'bookcase': 0.5,
	'beam': 0.1,
	'chair': 0.5,
	'column': 0.5,
	'door': 0.5,
	'sofa': 0.1,
	'table': 0.1,
	'window': 0.5,
	'ceiling': 0.01,
	'floor': 0.01,
	'wall': 0.01,
}

#color mapping for semantic segmentation
class_to_color_rgb = {
	0: (200,200,200), #clutter
	1: (0,100,100), #board
	2: (255,0,0), #bookcase
	3: (255,200,200), #beam
	4: (0,0,100), #chair
	5: (0,255,255), #column
	6: (0,100,0), #door
	7: (255,0,255), #sofa
	8: (50,50,50), #table
	9: (0,255,0), #window
	10: (255,255,0), #ceiling
	11: (0,0,255), #floor
	12: (255,165,0), #wall
}

#extend colors to NYU 40 classes
sample_state = numpy.random.RandomState(0)
for i in range(13, 41):
	class_to_color_rgb[i] = tuple(numpy.random.randint(0, 255, 3))
