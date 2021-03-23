# LRGNet: Learnable Region Growing for Point Cloud Segmentation

This repository contains code for the
[RAL paper](https://ieeexplore.ieee.org/document/9364361)
*LRGNet: Learnable Region Growing for Class-Agnostic Point Cloud Segmentation*.

![architecture](figures/architecture.png?raw=true)

## Prerequisites

1. numpy
2. scipy
3. scikit-learn
4. tensorflow
5. h5py
6. networkx

## Data Staging

Run the following script to download the necessary point cloud files in H5 format to the *data* folder.

```
bash download_data.sh
```

To use the Semantic KITTI dataset, follow the instructions on the [Semantic KITTI website](http://semantic-kitti.org/dataset.html)
and then run the following script:

```
python stage_semantic_kitti.py --dataset semantic_kitti/dataset/ --output data/kitti_train.h5 --sequences 00,01,02,03,04,05,06,07,09,10
python stage_semantic_kitti.py --dataset semantic_kitti/dataset/ --output data/kitti_val.h5 --sequences 08
```

## Data Visualization

To check the data shape/size contained in each H5 file:

```
python examine_h5.py data/scannet.h5
```

```
<HDF5 dataset "count_room": shape (312,), type "<i4">
<HDF5 dataset "points": shape (7924044, 8), type "<f4">
```

To convert the H5 data file into individual point cloud files (PLY) in format, run the script as follows.
PLY files can be opened using the [CloudCompare](https://www.danielgm.net/cc/) program

```bash
#Render the point clouds in original RGB color
python h5_to_ply.py data/s3dis_area3.h5 --rgb
#Render the point clouds colored according to segmentation ID
python h5_to_ply.py data/s3dis_area3.h5 --seg
```

```
...
Saved to data/viz/22.ply: (18464 points)
Saved to data/viz/23.ply: (20749 points)
```

To plot the instance color legend for a target room:

```bash
#Plot color legend for room #100 in ScanNet
python h5_to_ply.py data/scannet.h5 --target 100
```

## Benchmarks

Train benchmark networks such as PointNet and PointNet++ (pointnet2).
```bash
python train_pointnet.py --mode pointnet --train-area 1,2,3,4,6 --val-area 5
```

Run benchmark algorithms on each dataset. Mode is one of *normal*, *color*, *curvature*, *pointnet*, *pointnet2*, *edge*, *smoothness*, *fpfh*, *feature*.

```bash
python benchmarks.py --mode normal --area 5 --threshold 0.99 --save
```

Evaluate the cross-domain performance as follows:
```bash
python train_pointnet.py --mode pointnet --train-area scannet --val-area s3dis --cross-domain
python benchmarks.py --mode pointnet --train-area scannet --area s3dis --cross-domain
```

Evaluate the performance on the Semantic KITTI dataset as follows:
```bash
python benchmarks.py --mode feature --area kitti_val --resolution 0.3
python train_pointnet.py --mode pointnet --train-area kitti_train --val-area kitti_val
python benchmarks.py --mode pointnet --train-area kitti_train --area kitti_val --resolution 0.3
```

## Learn Region Grow (LRGNet)

Run region growing simulations to stage ground truth data for LRGNet.

```bash
python stage_data.py
#To apply data augmentation, run stage_data with different random seeds
for i in 0 1 2 3 4 5 6 7
do
	for j in s3dis scannet
	do
		python stage_data.py --seed $i --area $j
	done
done
```

Train LRGNet for each area of the S3DIS dataset.

```bash
python train_region_grow.py --train-area 1,2,3,4,6 --val-area 5
```

Test LrgNet and measure the accuracy metrics.

```bash
python test_region_grow.py --area 5 --save
python test_region_grow.py --area scannet --save
```

Test LRGNet using local search methods
```bash
python test_random_restart.py --area 5 --scoring ml
python test_random_restart.py --area 5 --scoring np
python test_beam_search.py --area 5 --scoring ml
python test_beam_search.py --area 5 --scoring np
```

Evaluate the cross-domain performance as follows:
```bash
python train_region_grow.py --train-area scannet --cross-domain
python test_region_grow.py --train-area scannet --area s3dis --cross-domain
python test_random_restart.py --train-area scannet --area s3dis --cross-domain --scoring np
```

Evaluate the performance on the Semantic KITTI dataset as follows:
```bash
for i in 01 02 03 04 05 06 07 09 10
do
    python stage_semantic_kitti.py --dataset semantic_kitti/dataset/ --output data/kitti_train_"$i".h5 --sequences $i --skip 1
done
for i in 0 1 2 3 4 5 6 7 9 10
do
    python stage_data.py --area kitti_train --resolution 0.3 --seed $i
done
python train_region_grow.py --train-area kitti_train --val-area kitti_val --multiseed 11 
python test_region_grow.py --area kitti_val --resolution 0.3 --save
```

## Results

Segmentation results on S3DIS dataset

![s3dis-results](figures/s3dis_results.png?raw=true)

Segmentation results on ScanNet dataset

![scannet-results](figures/scannet_results.png?raw=true)

Segmentation results on Semantic KITTI dataset

![kitti-results](figures/kitti_combined.png?raw=true)

Citation
--------

	@ARTICLE{chen2021ral,
		author={J. {Chen} and Z. {Kira} and Y. K. {Cho}},
		journal={IEEE Robotics and Automation Letters}, 
		title={LRGNet: Learnable Region Growing for Class-Agnostic Point Cloud Segmentation}, 
		year={2021},
		volume={6},
		number={2},
		pages={2799-2806},
		doi={10.1109/LRA.2021.3062607},
	}
