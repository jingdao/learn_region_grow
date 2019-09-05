# Learnable Region Growing for Point Cloud Segmentation

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
The *--rgb* flag will render the point clouds in original color whereas the *--seg* flag will color the point clouds according to segmentation ID.
PLY files can be opened using the [CloudCompare](https://www.danielgm.net/cc/) program

```
python h5_to_ply.py data/s3dis_area3.h5 --rgb
python h5_to_ply.py data/s3dis_area3.h5 --seg
```

