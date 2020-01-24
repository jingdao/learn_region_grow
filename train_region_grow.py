from learn_region_grow_util import *
import sys

BATCH_SIZE = 100
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
MAX_EPOCH = 100
VAL_STEP = 7
VAL_AREA = 1
FEATURE_SIZE = 13
MULTISEED = 5
initialized = False
for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = LrgNet(BATCH_SIZE, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
saver = tf.train.Saver()
MODEL_PATH = 'models/lrgnet_model%d.ckpt'%VAL_AREA

init = tf.global_variables_initializer()
sess.run(init, {})
for epoch in range(MAX_EPOCH):

	if not initialized or MULTISEED > 1:
		initialized = True
		train_inlier_points, train_inlier_count, train_neighbor_points, train_neighbor_count, train_add, train_remove, train_complete = [], [], [], [], [], [], []
		val_inlier_points, val_inlier_count, val_neighbor_points, val_neighbor_count, val_add, val_remove, val_complete = [], [], [], [], [], [], []

		for AREA in range(1,7):
			if MULTISEED > 0:
				SEED = epoch % MULTISEED
				f = h5py.File('data/multiseed/seed%d_area%d.h5'%(SEED,AREA),'r')
			else:
				f = h5py.File('data/staged_area%d.h5'%(AREA),'r')
			print('Loading %s ...'%f.filename)
			if AREA == VAL_AREA:
				val_complete.extend(f['complete'][:])
				count = f['count'][:]
				val_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					val_inlier_points.append(points[idp:idp+count[i], :])
					val_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:]
				val_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					val_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :])
					val_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
			else:
				train_complete.extend(f['complete'][:])
				count = f['count'][:]
				train_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					train_inlier_points.append(points[idp:idp+count[i], :])
					train_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:]
				train_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					train_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :])
					train_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
			if FEATURE_SIZE is None: 
				FEATURE_SIZE = points.shape[1]
			f.close()

		#filter out instances where the neighbor array is empty
		train_inlier_points = [train_inlier_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_inlier_count = [train_inlier_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_points = [train_neighbor_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_add = [train_add[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_remove = [train_remove[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_complete = [train_complete[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_complete = numpy.array(train_complete)
		train_neighbor_count = [train_neighbor_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		val_inlier_points = [val_inlier_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_inlier_count = [val_inlier_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_points = [val_neighbor_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_add = [val_add[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_remove = [val_remove[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_complete = [val_complete[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_complete = numpy.array(val_complete)
		val_neighbor_count = [val_neighbor_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		print('train',len(train_inlier_points),train_inlier_points[0].shape, len(train_neighbor_points), train_complete.shape)
		print('val',len(val_inlier_points),val_inlier_points[0].shape, len(val_neighbor_points), val_complete.shape)

	idx = numpy.arange(len(train_inlier_points))
	numpy.random.shuffle(idx)
	inlier_points = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT, FEATURE_SIZE))
	neighbor_points = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT, FEATURE_SIZE))
	input_add = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	input_remove = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT), dtype=numpy.int32)

	loss_arr = []
	add_prc_arr = []
	add_rcl_arr = []
	rmv_prc_arr = []
	rmv_rcl_arr = []
	cmp_prc_arr = []
	cmp_rcl_arr = []
	cmp_acc_arr = []
	num_batches = int(len(train_inlier_points) / BATCH_SIZE)
	for batch_id in range(num_batches):
		start_idx = batch_id * BATCH_SIZE
		end_idx = (batch_id + 1) * BATCH_SIZE
		for i in range(BATCH_SIZE):
			points_idx = idx[start_idx+i]
			N = train_inlier_count[points_idx]
			if N >= NUM_INLIER_POINT:
				subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
			else:
				subset = range(N) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
			inlier_points[i,:,:] = train_inlier_points[points_idx][subset, :]
			input_remove[i,:] = train_remove[points_idx][subset]
			N = train_neighbor_count[points_idx]
			if N >= NUM_NEIGHBOR_POINT:
				subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
			else:
				subset = range(N) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
			neighbor_points[i,:,:] = train_neighbor_points[points_idx][subset, :]
			input_add[i,:] = train_add[points_idx][subset]
		input_complete = train_complete[idx[start_idx:end_idx]]
		_, ls, ap, ar, rp, rr, cp, cr, ca = sess.run([net.train_op, net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl, net.completeness_prc, net.completeness_rcl, net.completeness_acc],
			{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.completeness_pl:input_complete, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})
		loss_arr.append(ls)
		add_prc_arr.append(ap)
		add_rcl_arr.append(ar)
		rmv_prc_arr.append(rp)
		rmv_rcl_arr.append(rr)
		cmp_prc_arr.append(cp)
		cmp_rcl_arr.append(cr)
		cmp_acc_arr.append(ca)
	print("Epoch %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f cmpl %.2f/%.2f/%.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(add_prc_arr),numpy.mean(add_rcl_arr),numpy.mean(rmv_prc_arr), numpy.mean(rmv_rcl_arr), numpy.mean(cmp_prc_arr), numpy.mean(cmp_rcl_arr), numpy.mean(cmp_acc_arr)))

	if epoch % VAL_STEP == VAL_STEP - 1:
		loss_arr = []
		add_prc_arr = []
		add_rcl_arr = []
		rmv_prc_arr = []
		rmv_rcl_arr = []
		cmp_prc_arr = []
		cmp_rcl_arr = []
		cmp_acc_arr = []
		num_batches = int(len(val_inlier_points) / BATCH_SIZE)
		for batch_id in range(num_batches):
			start_idx = batch_id * BATCH_SIZE
			end_idx = (batch_id + 1) * BATCH_SIZE
			for i in range(BATCH_SIZE):
				points_idx = start_idx+i
				N = val_inlier_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
				else:
					subset = range(N) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
				inlier_points[i,:,:] = val_inlier_points[points_idx][subset, :]
				input_remove[i,:] = val_remove[points_idx][subset]
				N = val_neighbor_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = range(N) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
				neighbor_points[i,:,:] = val_neighbor_points[points_idx][subset, :]
				input_add[i,:] = val_add[points_idx][subset]
			input_complete = val_complete[start_idx:end_idx]
			ls, ap, ar, rp, rr, cp, cr, ca = sess.run([net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl, net.completeness_prc, net.completeness_rcl, net.completeness_acc],
				{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.completeness_pl:input_complete, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove})
			loss_arr.append(ls)
			add_prc_arr.append(ap)
			add_rcl_arr.append(ar)
			rmv_prc_arr.append(rp)
			rmv_rcl_arr.append(rr)
			cmp_prc_arr.append(cp)
			cmp_rcl_arr.append(cr)
			cmp_acc_arr.append(ca)
		print("Validation %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f cmpl %.2f/%.2f/%.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(add_prc_arr),numpy.mean(add_rcl_arr),numpy.mean(rmv_prc_arr), numpy.mean(rmv_rcl_arr), numpy.mean(cmp_prc_arr), numpy.mean(cmp_rcl_arr), numpy.mean(cmp_acc_arr)))

saver.save(sess, MODEL_PATH)
