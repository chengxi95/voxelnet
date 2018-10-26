# `train.py`
main program to construct layers, training model, validating, predicting ,writing summary, saving it into file and tell the progress of the program

# `model`
## `model/model.py`
### `RPN3D`:
class tells how to construct layers and define training,validate,predict steps. When you create a RPN3D object, it will add a FeatureNet and a MiddleAndRPN layer into the graph and also multiple parameter controling and using in training,validating,predicting process.
### `train_step`:
a step in training process. Read data and return a dict that maps previous data onto new data. `output_feed` contains loss information

### `validate_step`:
almost same as train_step, except using data from `sample_test_data` which from validation_dir. When the number of the epoch of program is a integer times `summary_val_interval` (set to be 5 in default), the program will run `validata_step` and try to run `predict_step` if there are no error

### `predict_step`:
this function will plot boxes predicted by reading data from label on image


## `model/group_pointcloud.py`:

### `VFELayer`:
implementation of VFELayer decribe in paper. Containing a linear layer, batch normalization layer and Relu layer. Then the data is concatented with its own elementwise maxpooling feature

### `FeatureNet`:
containing two VFE layer, Feature.outputs give the Sparse Tensor Representation descirbe in paper after two VFE layers and elementwise maxpooling

## `model/rpn.py`
### `MideeleAndRPN`:
construct Convolutional Middle Layer and RPN describe in paper. self.loss is the loss using loss function in paper. self.delta_output is regression map and self.prob_output is probability score map

### `ConvMD`:
construct one convolutional middle layer as describe in paper. First parameter M means whether it is a 3d convolutional layer used in Convolutional Middle Layer or a 2d convolutional layer used in RPN. Batch normalization and Relu activation are also combined in this function

### `Deconv2D`:
construct Deconv2d layer describe in paper. It is part of RPN. Including batch normalization and relu activation


# `data/`
## `data/crop.py`:
### `main`
the main program will use calib the velodyne training data using matrix in its own calib file and then split the whole training set(velodyne data, label_2 data, calib data and image_2 data) 
into training set and validation set, based on the split file and form a structure folder we desire.

### `load_velodyne_points`:
read binary data from given velodyne file

### `load_calib`:
read $P_{rect}$, $R^{(0)}_{rect}$ and $T^{cam}_{velo}$ matrix from calib file in the given calib_dir, the difference between this and `load_calib` in `utlis/utils.py` is that $P_{rect}$ does not concatenate with [0,0,0,0], which means its shape is (3,4) rather than (4,4)

### `prepare_velo_points`:
replace the reflectance of velodyne points cloud s valude by 1, and transpose the array, so points can be directly multiplied by the camera's projection matrix

### `project_velo_points_in_imag`:
project 3d points into a 2d image

### `align_img_and_pc`:
using image file and calib file to calib raw velodyne data


# `utils/`
## `utils/kitti_loader`:
### `Processor`:
class for mulitiprocessing data augmentation

### `iterate_data`:
read data from given data_dir,augment the data if needed,return all information in data, inclouding tag, labels, vox_feature, vox_number, vox_coordinate, rgb and raw_lidar, they are all wrapper in an array ret

### `build_input`:
seperate voxel_dict_list get from mulitiprocessing augmentation and combine them into one feature, number and coordinate

## `utils/preprocess.py`:
### `process_pointcloud`:
transform points cloud into a (K,T,7) tensor where K is total number of voxel, T is maximum number of points a voxel can hold, 7 is the input encoding dimension for each point. This is record in `feature_buffer`,`coordinate_buffer` is a (K,3) array which stores the position for a voxel, `number_buffer` stores the number of points in each voxel. They are all wrapper in a dictionary voxel_dict

## `utils/data_aug.py`
### `aug_data`:
augment velodyne and label data if needed, you can set minimum number of points in a single velodyne file. The data will randomly choose one of the three data augmentation method mentioned in paper until its number is greater than the threshold you can. It will return the data's number `tag`, image data `rgb`,lidar date after augmentation `lidar_after_aug`,`voxel_dict` get from `lidar_after_aug` and label after augmentation `labebl_after_aug`

## `utils/utils.py`:
### `angle_in_limit`:
transform an `angle` into [$-\frac{\pi}{2},\frac{\pi}{2}$] by adding or subtracting number of $\pi$, if the difference between the input `angle` and $\frac{\pi}{2}$ is smaller than `limit_degree`, which is set to be $5^{\circ}$ in default

### `camera_to_lidar_point`:
transform the point's 3d coordinate (x,y,z) from carmera coordinate to lidar coordinate using $R^{(0)}_{rect}$ and $T^{cam}_{velo}$ matrix. If no matrix is given, it will use the average of the matrix in the training set

### `lidar_to_camera_point`:
transform the points's 3d coordinate(x,y,z) from lidar coordinate to camera coordinate using $R^{(0)}_{rect}$ and $T^{cam}_{velo}$ matrix. If no matrix is given, it will use the average of the matrix in the training set

### `camera_to_lidar_box`:
transform the box's 7d coordinate `(x, y, z, h, w, l, r)` from camera coordinate to lidar coordinate, using `camera_to_lidar_point` to transfrom the center location`(x, y, z)` 

### `lidar_to_camera_box`:
transform the box's 7d coordinate `(x, y, z, h, w, l, r)` from lidar coordinate to camera coordinate, using `lidar_to_camera_point` to transfrom the center location`(x, y, z)` 

### `corner_to_center_box3d`:
give coordinate of the box's 8 vertexes, return the coordinate of the center of box and h, w, l, ry/2

### `label_to_gt_box3d`:
transform the label (form by lines in a label_2 .txt file) into an (N,7) array, where N is the number of lines whose object is `cls`(car, pedestrain or cyclist),if `cls` is empty, return all lines. For each line, the data is arranged in `(x, y, z, h, w, l, r)`

### `cal_rpn_target`:
calculate IoU of feature_map

### `point_transform`:
transform the coordinate of n points after transition in (tx,ty,tz) direction and rotation around x,y,z axis with (rx, ry, rz) in radians

### `box_transform`:
transform the coordinate of n boxes (in (x, y, z, h, w, l, rz/y) after transition in (tx, ty, tz) and
rotation around x,y,z axis with (rx, ry ,rz) in radians

### `cal_anchors`:
return all anchor determined by the config in `config.py`, all anchors are return in shape (w,l,2,7)

### `cal_rpn_target`:
return positive and negative anchors using IoU of labels and anchor

### `cal_iou2d`:
calculate IoU of two 2d boxes in (x,y,,w,l,r) coordinate

### `cal_iou3d`:
calculate IoU of two 3d boxes in (x,y,z,h,w,l,r) coordinate





