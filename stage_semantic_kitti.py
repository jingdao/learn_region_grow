import numpy as np
import h5py
import sys
import os
import argparse
import imageio
import yaml
import itertools
import networkx as nx

parser = argparse.ArgumentParser("")
parser.add_argument( '--dataset', '-d', type=str, required=True, help='')
parser.add_argument( '--output', '-o', type=str, required=True, help='')
parser.add_argument( '--sequences', '-s', type=str, default='00,01,02,03,04,05,06,07,08,09,10', help='')
parser.add_argument( '--interval', '-i', type=int, default=10, help='')
parser.add_argument( '--min-cluster', '-m', type=int, default=50, help='')
parser.add_argument( '--voxel-resolution', '-v', type=float, default=0.3, help='')
parser.add_argument( '--downsample-resolution', '-r', type=float, default=0.05, help='')
FLAGS, unparsed = parser.parse_known_args()

def downsample(cloud, resolution):
    voxel_coordinates = [tuple(p) for p in np.round((cloud[:,:3] / resolution)).astype(int)]
    voxel_set = set()
    downsampled_cloud = []
    for i in range(len(cloud)):
        if not voxel_coordinates[i] in voxel_set:
            voxel_set.add(voxel_coordinates[i])
            downsampled_cloud.append(cloud[i])
    return np.array(downsampled_cloud)

# get class names
yaml_file = open(os.path.join(FLAGS.dataset, "semantic-kitti.yaml"), 'r')
config = yaml.full_load(yaml_file)
class_names = config['labels']
yaml_file.close()

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
    image_paths = os.path.join(FLAGS.dataset, "sequences",
                             sequence, "image_2")
    image_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(image_paths)) for f in fn]
    image_names.sort()

    rgb_map = {}
    stacked_points = []
    for offset in range(0, len(scan_names)):
        scan = np.fromfile(scan_names[offset], dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # get XYZ in world coordinate frame
        xyz_local = scan[:, 0:3]
        R_world_local = poses[offset][:3, :3]
        t_world_local = poses[offset][:3, 3]
        xyz_world = xyz_local.dot(R_world_local.T) + t_world_local
        xyz_voxels = [tuple(v) for v in np.round(xyz_world / FLAGS.voxel_resolution).astype(int)]
        
        # project RGB colors
        rgb = np.zeros((len(xyz_local), 3))
        image = imageio.imread(image_names[offset])
        xyz_homogenous = np.hstack((xyz_local, np.ones(len(xyz_local)).reshape(-1, 1)))
        xyz_homogenous = calib['P2'].dot(calib['Tr'].dot(xyz_homogenous.T)).T
        uv = np.round(xyz_homogenous[:, :2] / xyz_homogenous[:, 2:3]).astype(int)
        valid = xyz_homogenous[:, 2] > 0
        valid = np.logical_and(valid, uv[:, 0] >= 0)
        valid = np.logical_and(valid, uv[:, 0] < image.shape[1])
        valid = np.logical_and(valid, uv[:, 1] >= 0)
        valid = np.logical_and(valid, uv[:, 1] < image.shape[0])
        for i in np.arange(len(xyz_homogenous))[valid]:
            rgb[i, :] = image[uv[i,1], uv[i,0], :]
            if not xyz_voxels[i] in rgb_map:
                rgb_map[xyz_voxels[i]] = rgb[i, :]
        for i in np.arange(len(xyz_homogenous))[~valid]:
            if xyz_voxels[i] in rgb_map:
                rgb[i, :] = rgb_map[xyz_voxels[i]]
        rgb = rgb / 255.0 - 0.5

        # get point labels
        label = np.fromfile(label_names[offset], dtype=np.uint32)
        obj_id = [l >> 16 for l in label]
        cls_id = [l & 0xFFFF for l in label]

        # stack in Nx8 array
        points = np.zeros((len(xyz_world), 8))
        points[:, :3] = xyz_world
        points[:, 3:6] = rgb
        points[:, 6] = obj_id
        points[:, 7] = cls_id
        # filter out points with no valid color mapping
        points = points[~np.all(rgb == -0.5, axis=1), :]
        # filter out points from moving objects
        points = points[points[:, 7] < 250]
        stacked_points.extend(points)
        print('Processing %d points from %s'%(len(points), scan_names[offset][len(FLAGS.dataset):]))

        if offset % FLAGS.interval == FLAGS.interval - 1:
            stacked_points = np.array(stacked_points)
            stacked_points = downsample(stacked_points, FLAGS.downsample_resolution)

            # get equalized resolution for connected components
            equalized_idx = []
            unequalized_idx = []
            equalized_map = {}
            point_voxels = [tuple(v) for v in np.round(stacked_points[:,:3]/FLAGS.voxel_resolution).astype(int)]
            for i in range(len(stacked_points)):
                k = point_voxels[i]
                if not k in equalized_map:
                    equalized_map[k] = len(equalized_idx)
                    equalized_idx.append(i)
                unequalized_idx.append(equalized_map[k])
            points = stacked_points[equalized_idx, :]
            point_voxels = [tuple(v) for v in np.round(points[:,:3]/FLAGS.voxel_resolution).astype(int)]
            obj_id = points[:, 6]
            cls_id = points[:, 7]
            new_obj_id = np.zeros(len(obj_id), dtype=int)

            # connected components to label unassigned obj IDs
            original_obj_id = set(points[:, 6]) - set([0])
            cluster_id = 1
            for i in original_obj_id:
                new_obj_id[obj_id == i] = cluster_id
                cluster_id += 1 

            edges = []
            for i in range(len(point_voxels)):
                if obj_id[i] > 0:
                    continue
                k = point_voxels[i]
                for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
                    if offset!=(0,0,0):
                        kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                        if kk in equalized_map and cls_id[i] == cls_id[equalized_map[kk]]:
                            edges.append([i, equalized_map[kk]])
            G = nx.Graph(edges)
            clusters = nx.connected_components(G)
            clusters = [list(c) for c in clusters]
            for i in range(len(clusters)):
                if len(clusters[i]) > FLAGS.min_cluster:
                    new_obj_id[clusters[i]] = cluster_id
                    cluster_id += 1

            stacked_points[:, 6] = new_obj_id[unequalized_idx]
            stacked_points = stacked_points[stacked_points[:, 6] > 0, :]
            print('Creating data sample with %d->%d points %d->%d objects' % (len(stacked_points), len(points), len(original_obj_id), len(set(new_obj_id))))
            all_points.extend(stacked_points)
            count.append(len(stacked_points))
            stacked_points = []

h5_fout = h5py.File(FLAGS.output,'w')
h5_fout.create_dataset('points', data=all_points, compression='gzip', compression_opts=4, dtype=np.float32)
h5_fout.create_dataset('count_room', data=count, compression='gzip', compression_opts=4, dtype=np.int32)
h5_fout.close()

