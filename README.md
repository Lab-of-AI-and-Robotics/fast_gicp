
* Ref: https://github.com/SMRT-AIST/fast_gicp

- I add some useful functions for both cpp & python
- I modify gicp as it utilizes raw covariance by following normalized_ellipse mode (not plane mode), in order to meet the scales for multiple 3D pointclouds
=> scale = scale / scale[1] .max(1e-3)
  
* note that cov = R*S*(R*S)^T = R*SS*R^T, S = scale.asDiagonal();
* here, R = quaternion.toRotation();
* q = (q_x, q_y, q_z, q_w)
* R and SS can be obtained by SVD; R=U, scale**2 = singular_values.array()

Install for python
```shell
catkin_make -DCMAKE_BUILD_TYPE=Release
cd fast_gicp
python3 setup.py install --user
```

python usage (see src/fast_gicp/python):

```python
import pygicp

gicp = pygicp.FastGICP()

gicp.set_input_target(target)
gicp.set_input_source(source)

# set covariance from quaternion and scale by following normalized_elipse
nparray_of_quaternions = nparray_of_quaternions_Nx4.flatten()
nparray_of_scales = nparray_of_scales_NX3.flatten()
gicp.set_source_covariance_fromqs(nparray_of_quaternions, nparray_of_scales)
gicp.set_target_covariance_fromqs(nparray_of_quaternions, nparray_of_scales) => 0.002180 sec

# compute covariance by following normalized_elipse
calculate_source_covariance() # compute covariance from given input source pointcloud
calculate_target_covariance() # compute covariance from given input target pointcloud

correspondences, sq_distances = gicp.get_target_correspondence()
covariances = get_source_covariances()
covariances = get_target_covariances()
nparray_of_quaternions = get_source_rotationsq() => 0.00002277 sec
nparray_of_quaternions = get_target_rotationsq() 
nparray_of_scales = get_source_scales()          => 0.00002739 sec
nparray_of_scales = get_target_scales()
nparray_of_quaternions_Nx4 = np.reshape(nparray_of_quaternions, (-1,4))
nparray_of_scales_NX3 = np.reshape(nparray_of_scales, (-1,3))

```

EX) python using_previous_30.py dataset/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household tum 0.05 true
<img width="80%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/tum_30_elipse.png"/>
<img width="80%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/tum_30_elipse.gif"/>
