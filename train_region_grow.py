from learn_region_grow_util import *
import sys

BATCH_SIZE = 100
NUM_POINT = 256
NUM_CONTEXT_POINT = 32
MAX_EPOCH = 100
VAL_STEP = 10
VAL_AREA = 1
FEATURE_SIZE = 9
for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = LrgNet(BATCH_SIZE, NUM_POINT, FEATURE_SIZE, 7, NUM_CONTEXT_POINT)
saver = tf.train.Saver()
MODEL_PATH = 'models/lrgnet_model%d.ckpt'%VAL_AREA

train_points, train_count, train_neighbor_points, train_neighbor_count, train_class, train_complete = [], [], [], [], [], []
val_points, val_count, val_neighbor_points, val_neighbor_count, val_class, val_complete = [], [], [], [], [], []

for AREA in range(1,7):
	f = h5py.File('data/staged_area%d.h5'%AREA,'r')
	print('Loading %s ...'%f.filename)
	if AREA == VAL_AREA:
		val_complete.extend(f['complete'][:])
		count = f['count'][:]
		val_count.extend(count)
		points = f['points'][:]
		idp = 0
		for i in range(len(count)):
			val_points.append(points[idp:idp+count[i], :])
			idp += count[i]
		neighbor_count = f['neighbor_count'][:]
		val_neighbor_count.extend(neighbor_count)
		neighbor_points = f['neighbor_points'][:]
		neighbor_class = f['class'][:]
		idp = 0
		for i in range(len(neighbor_count)):
			val_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :])
			val_class.append(neighbor_class[idp:idp+neighbor_count[i]])
			idp += neighbor_count[i]
	else:
#	if AREA == VAL_AREA:
		train_complete.extend(f['complete'][:])
		count = f['count'][:]
		train_count.extend(count)
		points = f['points'][:]
		idp = 0
		for i in range(len(count)):
			train_points.append(points[idp:idp+count[i], :])
			idp += count[i]
		neighbor_count = f['neighbor_count'][:]
		train_neighbor_count.extend(neighbor_count)
		neighbor_points = f['neighbor_points'][:]
		neighbor_class = f['class'][:]
		idp = 0
		for i in range(len(neighbor_count)):
			train_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :])
			train_class.append(neighbor_class[idp:idp+neighbor_count[i]])
			idp += neighbor_count[i]
	f.close()

train_complete = numpy.array(train_complete)
val_complete = numpy.array(val_complete)
print('train',len(train_points),len(train_neighbor_points), train_complete.shape)
print('val',len(val_points),len(val_neighbor_points), val_complete.shape)

init = tf.global_variables_initializer()
sess.run(init, {})
for epoch in range(MAX_EPOCH):
	idx = numpy.arange(len(train_points))
	numpy.random.shuffle(idx)
	input_points = numpy.zeros((BATCH_SIZE, NUM_POINT, FEATURE_SIZE))
	context_points = numpy.zeros((BATCH_SIZE, 7, NUM_CONTEXT_POINT, FEATURE_SIZE))
	input_classes = numpy.zeros((BATCH_SIZE, 7, NUM_CONTEXT_POINT), dtype=numpy.int32)
	input_mask = numpy.zeros((BATCH_SIZE, 7, NUM_CONTEXT_POINT), dtype=numpy.int32)

	loss_arr = []
	cls_arr = []
	cmp_arr = []
	num_batches = int(len(train_points) / BATCH_SIZE)
	for batch_id in range(num_batches):
		start_idx = batch_id * BATCH_SIZE
		end_idx = (batch_id + 1) * BATCH_SIZE
		for i in range(BATCH_SIZE):
			points_idx = idx[start_idx+i]
			subset = numpy.random.choice(train_count[points_idx], NUM_POINT, replace=train_count[points_idx]<NUM_POINT)
			input_points[i,:,:] = train_points[points_idx][subset, :]
			for j in range(7):
				if train_neighbor_count[points_idx*7+j]>0:
					subset = numpy.random.choice(train_neighbor_count[points_idx*7+j], NUM_CONTEXT_POINT, replace=train_neighbor_count[points_idx*7+j]<NUM_CONTEXT_POINT)
					context_points[i,j,:,:] = train_neighbor_points[points_idx*7+j][subset, :]
					input_classes[i,j,:] = train_class[points_idx*7+j][subset]
					input_mask[i,j,:] = 1
				else:
					input_mask[i,j,:] = 0
		input_complete = train_complete[idx[start_idx:end_idx]]
		_, ls, cls, cmpl = sess.run([net.train_op, net.loss, net.class_acc, net.completeness_acc],
			{net.input_pl:input_points, net.context_pl:context_points, net.mask_pl:input_mask, net.completeness_pl:input_complete, net.class_pl:input_classes})
		loss_arr.append(ls)
		cls_arr.append(cls)
		cmp_arr.append(cmpl)
	print("Epoch %d loss %.2f cls %.3f cmpl %.3f"%(epoch,numpy.mean(loss_arr),numpy.mean(cls_arr),numpy.mean(cmp_arr)))

	if epoch % VAL_STEP == VAL_STEP - 1:
		loss_arr = []
		cls_arr = []
		cmp_arr = []
		num_batches = int(len(val_points) / BATCH_SIZE)
		for batch_id in range(num_batches):
			start_idx = batch_id * BATCH_SIZE
			end_idx = (batch_id + 1) * BATCH_SIZE
			for i in range(BATCH_SIZE):
				points_idx = start_idx+i
				subset = numpy.random.choice(val_count[points_idx], NUM_POINT, replace=val_count[points_idx]<NUM_POINT)
				input_points[i,:,:] = val_points[points_idx][subset, :]
				for j in range(7):
					if val_neighbor_count[points_idx*7+j]>0:
						subset = numpy.random.choice(val_neighbor_count[points_idx*7+j], NUM_CONTEXT_POINT, replace=val_neighbor_count[points_idx*7+j]<NUM_CONTEXT_POINT)
						context_points[i,j,:,:] = val_neighbor_points[points_idx*7+j][subset, :]
						input_classes[i,j,:] = val_class[points_idx*7+j][subset]
						input_mask[i,j,:] = True
					else:
						input_mask[i,j,:] = False
			input_complete = val_complete[start_idx:end_idx]
			ls, cls, cmpl = sess.run([net.loss, net.class_acc, net.completeness_acc],
				{net.input_pl:input_points, net.context_pl:context_points, net.mask_pl:input_mask, net.completeness_pl:input_complete, net.class_pl:input_classes})
			loss_arr.append(ls)
			cls_arr.append(cls)
			cmp_arr.append(cmpl)
		print("Validation %d loss %.2f cls %.3f cmpl %.3f"%(epoch,numpy.mean(loss_arr),numpy.mean(cls_arr),numpy.mean(cmp_arr)))

saver.save(sess, MODEL_PATH)