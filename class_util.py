import numpy

#list of all classes
classes = ['clutter', 'board', 'bookcase', 'beam', 'chair', 'column', 'door', 'sofa', 'table', 'window', 'ceiling', 'floor', 'wall']
classes_s3dis = ['clutter', 'board', 'bookcase', 'beam', 'chair', 'column', 'door', 'sofa', 'table', 'window', 'ceiling', 'floor', 'wall']
classes_nyu40 = ['none','wall','floor','cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
                'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refrigerator',
                'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
classes_kitti = [''] * 260
classes_kitti[0] = "unlabeled"
classes_kitti[1] = "outlier"
classes_kitti[10] = "car"
classes_kitti[11] = "bicycle"
classes_kitti[13] = "bus"
classes_kitti[15] = "motorcycle"
classes_kitti[16] = "on-rails"
classes_kitti[18] = "truck"
classes_kitti[20] = "other-vehicle"
classes_kitti[30] = "person"
classes_kitti[31] = "bicyclist"
classes_kitti[32] = "motorcyclist"
classes_kitti[40] = "road"
classes_kitti[44] = "parking"
classes_kitti[48] = "sidewalk"
classes_kitti[49] = "other-ground"
classes_kitti[50] = "building"
classes_kitti[51] = "fence"
classes_kitti[52] = "other-structure"
classes_kitti[60] = "lane-marking"
classes_kitti[70] = "vegetation"
classes_kitti[71] = "trunk"
classes_kitti[72] = "terrain"
classes_kitti[80] = "pole"
classes_kitti[81] = "traffic-sign"
classes_kitti[99] = "other-object"
classes_kitti[252] = "moving-car"
classes_kitti[253] = "moving-bicyclist"
classes_kitti[254] = "moving-person"
classes_kitti[255] = "moving-motorcyclist"
classes_kitti[256] = "moving-on-rails"
classes_kitti[257] = "moving-bus"
classes_kitti[258] = "moving-truck"
classes_kitti[259] = "moving-other-vehicle"

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

#extend colors to larger number of classes
sample_state = numpy.random.RandomState(0)
for i in range(13, max(len(classes_s3dis), len(classes_nyu40), len(classes_kitti))):
	class_to_color_rgb[i] = tuple(sample_state.randint(0, 255, 3))
