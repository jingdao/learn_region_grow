from learn_region_grow_util import *
import sys

BATCH_SIZE = 5
NUM_INLIER_POINT = 256
NUM_NEIGHBOR_POINT = 256
SEQ_LEN = 100
MAX_EPOCH = 100
VAL_STEP = 7
VAL_AREA = 1
FEATURE_SIZE = 13
MULTISEED = 0
initialized = False
for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = LrgNet(BATCH_SIZE, SEQ_LEN, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
saver = tf.train.Saver()
MODEL_PATH = 'models/lrgnet_rnn_model%d.ckpt'%VAL_AREA

init = tf.global_variables_initializer()
sess.run(init, {})
for epoch in range(MAX_EPOCH):

	if not initialized or MULTISEED > 1:
		initialized = True
		train_inlier_points, train_inlier_count, train_neighbor_points, train_neighbor_count, train_add, train_remove, train_complete, train_steps, train_cum_steps = [], [], [], [], [], [], [], [], []
		val_inlier_points, val_inlier_count, val_neighbor_points, val_neighbor_count, val_add, val_remove, val_complete = [], [], [], [], [], [], []

		for AREA in range(1,7):
			if MULTISEED > 0:
				SEED = epoch % MULTISEED
				f = h5py.File('data/multiseed/seed%d_area%d.h5'%(SEED,AREA),'r')
			else:
#				f = h5py.File('data/staged_area%d.h5'%(AREA),'r')
				f = h5py.File('data/small_area%d.h5'%(AREA),'r')
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
				val_steps = f['steps'][:]
				val_cum_steps = numpy.cumsum(val_steps) - val_steps
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
				ts = f['steps'][:]
				train_steps.extend(ts)
				train_cum_steps.extend(numpy.cumsum(ts) - ts)
			if FEATURE_SIZE is None: 
				FEATURE_SIZE = points.shape[1]
			f.close()

		#filter out instances where the neighbor array is empty
		train_complete = numpy.array(train_complete)
		val_complete = numpy.array(val_complete)
		print('train',len(train_inlier_points),train_inlier_points[0].shape, len(train_neighbor_points), train_complete.shape)
		print('val',len(val_inlier_points),val_inlier_points[0].shape, len(val_neighbor_points), val_complete.shape)

	idx = numpy.arange(len(train_steps))
	numpy.random.shuffle(idx)
	inlier_points = numpy.zeros((BATCH_SIZE*SEQ_LEN, NUM_INLIER_POINT, FEATURE_SIZE))
	neighbor_points = numpy.zeros((BATCH_SIZE*SEQ_LEN, NUM_NEIGHBOR_POINT, FEATURE_SIZE))
	input_add = numpy.zeros((BATCH_SIZE*SEQ_LEN, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	input_remove = numpy.zeros((BATCH_SIZE*SEQ_LEN, NUM_INLIER_POINT), dtype=numpy.int32)
	input_complete = numpy.zeros(BATCH_SIZE*SEQ_LEN, dtype=numpy.float32)
	input_seq = numpy.zeros(BATCH_SIZE, dtype=numpy.int32)
	input_seq_mask = numpy.zeros(BATCH_SIZE*SEQ_LEN, dtype=numpy.bool)

	loss_arr = []
	add_prc_arr = []
	add_rcl_arr = []
	rmv_prc_arr = []
	rmv_rcl_arr = []
	cmp_prc_arr = []
	cmp_rcl_arr = []
	cmp_acc_arr = []
	num_batches = int(len(train_steps) / BATCH_SIZE)
	for A in range(num_batches):
		for B in range(BATCH_SIZE):
			batch_id = A*BATCH_SIZE + B
			current_step = min(train_steps[idx[batch_id]], SEQ_LEN)
			step_offset = train_cum_steps[idx[batch_id]]
			for i in range(current_step):
				points_idx = step_offset + i
				N = train_inlier_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
				else:
					subset = range(N) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
				inlier_points[B*SEQ_LEN+i,:,:] = train_inlier_points[points_idx][subset, :]
				input_remove[B*SEQ_LEN+i,:] = train_remove[points_idx][subset]
				N = train_neighbor_count[points_idx]
				if N >= NUM_NEIGHBOR_POINT:
					subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = range(N) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
				neighbor_points[B*SEQ_LEN+i,:,:] = train_neighbor_points[points_idx][subset, :]
				input_add[B*SEQ_LEN+i,:] = train_add[points_idx][subset]
			input_complete[B*SEQ_LEN:B*SEQ_LEN+current_step] = train_complete[step_offset:step_offset+current_step]
			input_seq[B] = current_step
			input_seq_mask[B*SEQ_LEN:(B+1)*SEQ_LEN] = False
			input_seq_mask[B*SEQ_LEN:B*SEQ_LEN+current_step] = True
		_, ls, ap, ar, rp, rr, cp, cr, ca= sess.run([net.train_op, net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl, net.completeness_prc, net.completeness_rcl, net.completeness_acc],
			{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.completeness_pl:input_complete, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove, net.seq_pl:input_seq, net.seq_mask_pl:input_seq_mask})
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
		num_batches = int(len(val_steps) / BATCH_SIZE)
		for A in range(num_batches):
			for B in range(BATCH_SIZE):
				batch_id = A*BATCH_SIZE + B
				current_step = min(val_steps[batch_id], SEQ_LEN)
				step_offset = val_cum_steps[batch_id]
				for i in range(current_step):
					points_idx = step_offset+i
					N = val_inlier_count[points_idx]
					if N >= NUM_INLIER_POINT:
						subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
					else:
						subset = range(N) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
					inlier_points[B*SEQ_LEN+i,:,:] = val_inlier_points[points_idx][subset, :]
					input_remove[B*SEQ_LEN+i,:] = val_remove[points_idx][subset]
					N = val_neighbor_count[points_idx]
					if N >= NUM_INLIER_POINT:
						subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
					else:
						subset = range(N) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
					neighbor_points[B*SEQ_LEN+i,:,:] = val_neighbor_points[points_idx][subset, :]
					input_add[B*SEQ_LEN+i,:] = val_add[points_idx][subset]
				input_complete[B*SEQ_LEN:B*SEQ_LEN+current_step] = val_complete[step_offset:step_offset+current_step]
				input_seq[B] = current_step
				input_seq_mask[B*SEQ_LEN:(B+1)*SEQ_LEN] = False
				input_seq_mask[B*SEQ_LEN:B*SEQ_LEN+current_step] = True
			ls, ap, ar, rp, rr, cp, cr, ca = sess.run([net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl, net.completeness_prc, net.completeness_rcl, net.completeness_acc],
				{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.completeness_pl:input_complete, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove, net.seq_pl:input_seq, net.seq_mask_pl:input_seq_mask})
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
