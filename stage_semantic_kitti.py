import numpy as np
import h5py
import sys
import os
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument( '--dataset', '-d', type=str, required=True, help='')
parser.add_argument( '--output', '-o', type=str, required=True, help='')
parser.add_argument( '--sequences', '-s', type=str, default='00,01,02,03,04,05,06,07,08,09,10', help='')
parser.add_argument( '--interval', '-i', type=int, default=10, help='')
FLAGS, unparsed = parser.parse_known_args()

all_points = []
count = []
for sequence in FLAGS.sequences.split(','):
    # get camera calibration
    calib_file = open(os.path.join(FLAGS.dataset, "sequences", sequence, "calib.txt"), 'r')
    calib = {}
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()

    # get poses
    poses = []
    Tr = calib["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    pose_file = open(os.path.join(FLAGS.dataset, "sequences", sequence, "poses.txt"), 'r')
    for line in pose_file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    pose_file.close()

    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            sequence, "velodyne")
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()
    label_paths = os.path.join(FLAGS.dataset, "sequences",
                             sequence, "labels")
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    for offset in range(0, len(scan_names), FLAGS.interval):
        scan = np.fromfile(scan_names[offset], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        xyz = scan[:, 0:3]
        rotation = poses[offset][:3, :3]
        translation = poses[offset][:3, 3]
        xyz = xyz.dot(rotation.T) + translation
        label = np.fromfile(label_names[offset], dtype=np.uint32)
        obj_id = [l >> 16 for l in label]
        cls_id = [l & 0xFFFF for l in label]
        rgb = np.zeros((len(xyz), 3))
        points = np.zeros((len(xyz), 8))
        points[:, :3] = xyz
        points[:, 3:6] = rgb
        points[:, 6] = obj_id
        points[:, 7] = cls_id
        print('Processing %d points from %s'%(len(points), scan_names[offset]))
        all_points.extend(points)
        count.append(len(points))

h5_fout = h5py.File(FLAGS.output,'w')
h5_fout.create_dataset('points', data=all_points, compression='gzip', compression_opts=4, dtype=np.float32)
h5_fout.create_dataset('count_room', data=count, compression='gzip', compression_opts=4, dtype=np.int32)
h5_fout.close()

