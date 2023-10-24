
* Ref: https://github.com/SMRT-AIST/fast_gicp
* Modified by Hyeonwoo Yu, UNIST

- I add some useful functions for both cpp & python
- I modify gicp as it utilizes raw covariance by following normalized_ellipse mode (not plane mode),
- in order to meet the scales for multiple 3D pointclouds
- => scale = scale / scale[1] .max(1e-3)
  
* note that cov = R*S*(R*S)^T = R*SS*R^T, S = scale.asDiagonal();
* here, R = quaternion.toRotation();
* q = (q_x, q_y, q_z, q_w)
* R and SS can be obtained by SVD; R=U, scale**2 = singular_values.array()

python usage (see src/fast_gicp/python):

```python
import pygicp

gicp = pygicp.FastGICP()

gicp.set_input_target(target)
gicp.set_input_source(source)

# set covariance with raw covariance (it does not normalize anything)
set_source_covariance_from6(array_of_array[cov00, cov01, cov02, cov11, cov12, cov22])
set_target_covariance_from6(array_of_array[cov00, cov01, cov02, cov11, cov12, cov22])
set_source_covariance_from3x3( array_of_mat3x3)
set_target_covariance_from3x3( array_of_mat3x3)

# set covariance from quaternion and scale by following normalized_elipse
gicp.set_source_covariance_fromqs(list_of_quaternions, list_of_scales)
gicp.set_target_covariance_fromqs(list_of_quaternions, list_of_scales)

# compute covariance by following normalized_elipse
calculate_source_covariance() # compute covariance from given input source pointcloud
calculate_target_covariance() # compute covariance from given input target pointcloud

correspondences, sq_distances = gicp.get_target_correspondence()
covariances = get_source_covariances()
covariances = get_target_covariances()
rotations_quaternion = get_source_rotationsq()
rotations_quaternion = get_target_rotationsq()
scales = get_source_scales()
target = get_source_target()

```

