
# I modify gicp as it uses raw covariance as default (not plane mode)

python usage:

import pygicp

gicp = pygicp.FastGICP()

gicp.set_input_target(target)
gicp.set_input_source(source)
set_source_covariance_from6(array_of_array[cov00, cov01, cov02, cov11, cov12, cov22])
set_target_covariance_from6(array_of_array[cov00, cov01, cov02, cov11, cov12, cov22])
set_source_covariance_from3x3( array_of_mat3x3)
set_target_covariance_from3x3( array_of_mat3x3)
calculate_source_covariance() # compute covariance from given input source pointcloud
calculate_target_covariance() # compute covariance from given input target pointcloud
correspondences, sq_distances = gicp.get_target_correspondence()
covariances = get_source_covariances()
covariances = get_target_covariances()
rotations_quaternion = get_source_rotationsq()
rotations_quaternion = get_target_rotationsq()
scales = get_source_scales()
target = get_source_target()

