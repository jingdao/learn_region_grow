# LRGNet: Learnable Region Growing for Point Cloud Segmentation

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

## Benchmarks

Train benchmark networks such as PointNet and PointNet++ (pointnet2).
```bash
for i in 1 2 3 4 5 6
do
	python -u train_pointnet.py --mode pointnet --area $i >> models/log_pointnet_model$i.txt
done
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

## Learn Region Grow (LRGNet)

Run region growing simulations to stage ground truth data for LRGNet.

```bash
python stage_data.py
#To apply data augmentation, run stage_data with different random seeds
for i in 0 1 2 3 4 5 6 7
do
	python stage_data.py --seed $i
done
```

Train LRGNet for each area of the S3DIS dataset.

```bash
for i in 1 2 3 4 5 6
do
	python train_region_grow.py --area $i >> models/log_lrgnet_model$i.txt
done
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

## Results

Segmentation results on S3DIS dataset

![s3dis-results](figures/s3dis_results.png?raw=true)

Segmentation results on ScanNet dataset

![scannet-results](figures/scannet_results.png?raw=true)

