#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <boost/filesystem.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

namespace py = pybind11;

fast_gicp::NeighborSearchMethod search_method(const std::string& neighbor_search_method) {
  if(neighbor_search_method == "DIRECT1") {
    return fast_gicp::NeighborSearchMethod::DIRECT1;
  } else if (neighbor_search_method == "DIRECT7") {
    return fast_gicp::NeighborSearchMethod::DIRECT7;
  } else if (neighbor_search_method == "DIRECT27") {
    return fast_gicp::NeighborSearchMethod::DIRECT27;
  } else if (neighbor_search_method == "DIRECT_RADIUS") {
    return fast_gicp::NeighborSearchMethod::DIRECT_RADIUS;
  }

  std::cerr << "error: unknown neighbor search method " << neighbor_search_method << std::endl;
  return fast_gicp::NeighborSearchMethod::DIRECT1;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr eigen2pcl(const Eigen::Matrix<double, -1, 3>& points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->resize(points.rows());

  for(int i=0; i<points.rows(); i++) {
    cloud->at(i).getVector3fMap() = points.row(i).cast<float>();
  }
  return cloud;
}

Eigen::Matrix<double, -1, 3> downsample(const Eigen::Matrix<double, -1, 3>& points, double downsample_resolution) {
  auto cloud = eigen2pcl(points);

  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  voxelgrid.setInputCloud(cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  voxelgrid.filter(*filtered);

  Eigen::Matrix<float, -1, 3> filtered_points(filtered->size(), 3);
  for(int i=0; i<filtered->size(); i++) {
    filtered_points.row(i) = filtered->at(i).getVector3fMap();
  }

  return filtered_points.cast<double>();
}

Eigen::Matrix4d align_points(
  const Eigen::Matrix<double, -1, 3>& target,
  const Eigen::Matrix<double, -1, 3>& source,
  const std::string& method,
  double downsample_resolution,
  int k_correspondences,
  double max_correspondence_distance,
  double voxel_resolution,
  int num_threads,
  const std::string& neighbor_search_method,
  double neighbor_search_radius,
  const Eigen::Matrix4f& initial_guess
) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = eigen2pcl(target);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = eigen2pcl(source);

  if(downsample_resolution > 0.0) {
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*filtered);
    target_cloud.swap(filtered);

    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*filtered);
    source_cloud.swap(filtered);
  }

  std::shared_ptr<fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>> reg;

  if(method == "GICP") {
    std::shared_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>);
    gicp->setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp->setCorrespondenceRandomness(k_correspondences);
    gicp->setNumThreads(num_threads);
    reg = gicp;
  } else if (method == "VGICP") {
    std::shared_ptr<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setResolution(voxel_resolution);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method));
    vgicp->setNumThreads(num_threads);
    reg = vgicp;
  } else if (method == "VGICP_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    vgicp->setResolution(voxel_resolution);
    reg = vgicp;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else if (method == "NDT_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>> ndt(new fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>);
    ndt->setResolution(voxel_resolution);
    ndt->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    reg = ndt;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else {
    std::cerr << "error: unknown registration method " << method << std::endl;
    return Eigen::Matrix4d::Identity();
  }

  reg->setInputTarget(target_cloud);
  reg->setInputSource(source_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  reg->align(*aligned, initial_guess);

  return reg->getFinalTransformation().cast<double>();
}

using LsqRegistration = fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
using FastGICP = fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
using FastVGICP = fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
#ifdef USE_VGICP_CUDA
using FastVGICPCuda = fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>;
using NDTCuda = fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
#endif

PYBIND11_MODULE(pygicp, m) {
  m.def("downsample", &downsample, "downsample points");

  m.def("align_points", &align_points, "align two point sets",
    py::arg("target"),
    py::arg("source"),
    py::arg("method") = "GICP",
    py::arg("downsample_resolution") = -1.0,
    py::arg("k_correspondences") = 15,
    py::arg("max_correspondence_distance") = std::numeric_limits<double>::max(),
    py::arg("voxel_resolution") = 1.0,
    py::arg("num_threads") = 0,
    py::arg("neighbor_search_method") = "DIRECT1",
    py::arg("neighbor_search_radius") = 1.5,
    py::arg("initial_guess") = Eigen::Matrix4f::Identity()
  );

  py::class_<LsqRegistration, std::shared_ptr<LsqRegistration>>(m, "LsqRegistration")
    .def("set_input_target", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("swap_source_and_target", &LsqRegistration::swapSourceAndTarget)
    .def("get_final_hessian", &LsqRegistration::getFinalHessian)
    .def("get_final_transformation", &LsqRegistration::getFinalTransformation)
    .def("get_fitness_score", [] (LsqRegistration& reg, const double max_range) { return reg.getFitnessScore(max_range); })
    .def("align",
      [] (LsqRegistration& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;

  py::class_<FastGICP, LsqRegistration, std::shared_ptr<FastGICP>>(m, "FastGICP")
    .def(py::init())
    .def("set_num_threads", &FastGICP::setNumThreads)
    .def("set_correspondence_randomness", &FastGICP::setCorrespondenceRandomness)
    .def("set_max_correspondence_distance", &FastGICP::setMaxCorrespondenceDistance)
    .def("get_source_covariances", [] (FastGICP& gicp){
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs = gicp.getSourceCovariances();
      py::list data; for(const auto& c:source_covs) data.append(c); return data;})
    .def("get_target_covariances", [] (FastGICP& gicp){
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs = gicp.getTargetCovariances();
      py::list data; for(const auto& c:target_covs) data.append(c); return data;})
    .def("get_source_rotationsq", [] (FastGICP& gicp){
      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> source_rotationsq = gicp.getSourceRotationsq();
      py::list data; for(const auto& q:source_rotationsq) data.append(q); return data;})
    .def("get_target_rotationsq", [] (FastGICP& gicp){
      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> target_rotationsq = gicp.getTargetRotationsq();
      py::list data; for(const auto& q:target_rotationsq) data.append(q); return data;})
    .def("get_source_scales", [] (FastGICP& gicp){
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> source_scales = gicp.getSourceScales();
      py::list data; for(const auto& s:source_scales) data.append(s); return data;})
    .def("get_target_scales", [] (FastGICP& gicp){
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_scales = gicp.getTargetScales();
      py::list data; for(const auto& s:target_scales) data.append(s); return data;})
    .def("calculate_source_covariance", &FastGICP::calculateSourceCovariance)
    .def("calculate_target_covariance", &FastGICP::calculateTargetCovariance)
    .def("get_target_correspondence", [] (FastGICP& gicp){
    	py::list correspondences = py::cast(gicp.getTargetCorrespondences());
    	py::list sq_distances = py::cast(gicp.getTargetSqDistances());
    	return py::make_tuple(correspondences, sq_distances);
    })
    .def("set_source_covariances_fromqs", [] (FastGICP& gicp, py::list listOfSourceQ, py::list listOfScales){
    	int size = py::len(listOfSourceQ);
    	if(size!=py::len(listOfScales)){ std::cerr<<"size not matched" <<std::endl; return;}
    	const auto input_rotationsq = listOfSourceQ.cast<std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>>();
    	const auto input_scales = listOfScales.cast<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>>();
    	gicp.setSourceCovariances(input_rotationsq, input_scales);
    })
    .def("set_target_covariances_fromqs", [] (FastGICP& gicp, py::list listOfSourceQ, py::list listOfScales){
    	int size = py::len(listOfSourceQ);
    	if(size!=py::len(listOfScales)){ std::cerr<<"size not matched" <<std::endl; return;}
    	const auto input_rotationsq = listOfSourceQ.cast<std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>>();
    	const auto input_scales = listOfScales.cast<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>>();
    	gicp.setTargetCovariances(input_rotationsq, input_scales);
    })
    .def("set_source_covariances_from6", [] (FastGICP& gicp, py::list listOfSourceCovariancesNumpy){
      /** assume we have something like:
      [ np.array(cov1[0][0], cov1[0][1], cov1[0][2], cov1[1][1], cov1[1][2], cov1[2][2]),
       np.array(cov2[0][0], cov2[0][1], cov2[0][2], cov2[1][1], cov2[1][2], cov2[2][2]),
       ...
       ]
       which has upper commonents of the covariance.
      **/
      // 2D numpy array to c++
      // ref: https://www.appsloveworld.com/cplus/100/589/howto-pass-a-list-of-numpy-array-with-pybind?expand_article=1
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> sourceCovariances;
      for (py::handle covarianceNumpyArray: listOfSourceCovariancesNumpy){
        Eigen::Matrix4d covMat = Eigen::Matrix4d::Zero();
        py::array_t<double> covariance = py::cast<py::array>(covarianceNumpyArray);
        auto requestedCovariance = covariance.request();
        double* pblockedCovariance = (double*) requestedCovariance.ptr;
        covMat(0,0) = pblockedCovariance[0];
        covMat(0,1) = pblockedCovariance[1];
        covMat(0,2) = pblockedCovariance[2];
        covMat(1,0) = pblockedCovariance[1];
        covMat(1,1) = pblockedCovariance[3];
        covMat(1,2) = pblockedCovariance[4];
        covMat(2,0) = pblockedCovariance[2];
        covMat(2,1) = pblockedCovariance[4];
        covMat(2,2) = pblockedCovariance[5];
        sourceCovariances.push_back(covMat);
      }
      gicp.setSourceCovariances(sourceCovariances);
    })
    .def("set_target_covariances_from6", [] (FastGICP& gicp, py::list listOfTargetCovariancesNumpy){
      /** assume we have something like:
      [ np.array(cov1[0][0], cov1[0][1], cov1[0][2], cov1[1][1], cov1[1][2], cov1[2][2]),
       np.array(cov2[0][0], cov2[0][1], cov2[0][2], cov2[1][1], cov2[1][2], cov2[2][2]),
       ...
       ]
       which has upper commonents of the covariance.
      **/
      // 2D numpy array to c++
      // ref: https://www.appsloveworld.com/cplus/100/589/howto-pass-a-list-of-numpy-array-with-pybind?expand_article=1
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> targetCovariances;
      for (py::handle covarianceNumpyArray: listOfTargetCovariancesNumpy){
        Eigen::Matrix4d covMat = Eigen::Matrix4d::Zero();
        py::array_t<double> covariance = py::cast<py::array>(covarianceNumpyArray);
        auto requestedCovariance = covariance.request();
        double* pblockedCovariance = (double*) requestedCovariance.ptr;
        covMat(0,0) = pblockedCovariance[0];
        covMat(0,1) = pblockedCovariance[1];
        covMat(0,2) = pblockedCovariance[2];
        covMat(1,0) = pblockedCovariance[1];
        covMat(1,1) = pblockedCovariance[3];
        covMat(1,2) = pblockedCovariance[4];
        covMat(2,0) = pblockedCovariance[2];
        covMat(2,1) = pblockedCovariance[4];
        covMat(2,2) = pblockedCovariance[5];
        targetCovariances.push_back(covMat);
      }
      gicp.setTargetCovariances(targetCovariances);
    })
    .def("set_source_covariances_from3x3", [] (FastGICP& gicp, py::list listOfSourceCovariancesNumpy){
      /** assume we have something like:
      [ np.array(cov1_mat3x3),
       np.array(cov2_mat3x3),
       ...
       ]
      **/
      // 2D numpy array to c++
      // ref: https://www.appsloveworld.com/cplus/100/589/howto-pass-a-list-of-numpy-array-with-pybind?expand_article=1
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> sourceCovariances;
      for (py::handle covarianceNumpyArray: listOfSourceCovariancesNumpy){
        Eigen::Matrix4d covMat = Eigen::Matrix4d::Zero();
        py::array_t<double> covariance = py::cast<py::array>(covarianceNumpyArray);
        auto requestedCovariance = covariance.request();
        double* pblockedCovariance = (double*) requestedCovariance.ptr;
        covMat(0,0) = pblockedCovariance[3*0+0];
        covMat(0,1) = pblockedCovariance[3*0+1];
        covMat(0,2) = pblockedCovariance[3*0+2];
        covMat(1,0) = pblockedCovariance[3*1+0];
        covMat(1,1) = pblockedCovariance[3*1+1];
        covMat(1,2) = pblockedCovariance[3*1+2];
        covMat(2,0) = pblockedCovariance[3*2+0];
        covMat(2,1) = pblockedCovariance[3*2+1];
        covMat(2,2) = pblockedCovariance[3*2+2];
        sourceCovariances.push_back(covMat);
      }
      gicp.setSourceCovariances(sourceCovariances);
    })
    .def("set_target_covariances_from3x3", [] (FastGICP& gicp, py::list listOftargetCovariancesNumpy){
      /** assume we have something like:
      [ np.array(cov1_mat3x3),
       np.array(cov2_mat3x3),
       ...
       ]
      **/
      // 2D numpy array to c++
      // ref: https://www.appsloveworld.com/cplus/100/589/howto-pass-a-list-of-numpy-array-with-pybind?expand_article=1
      std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> targetCovariances;
      for (py::handle covarianceNumpyArray: listOftargetCovariancesNumpy){
        Eigen::Matrix4d covMat = Eigen::Matrix4d::Zero();
        py::array_t<double> covariance = py::cast<py::array>(covarianceNumpyArray);
        auto requestedCovariance = covariance.request();
        double* pblockedCovariance = (double*) requestedCovariance.ptr;
        covMat(0,0) = pblockedCovariance[3*0+0];
        covMat(0,1) = pblockedCovariance[3*0+1];
        covMat(0,2) = pblockedCovariance[3*0+2];
        covMat(1,0) = pblockedCovariance[3*1+0];
        covMat(1,1) = pblockedCovariance[3*1+1];
        covMat(1,2) = pblockedCovariance[3*1+2];
        covMat(2,0) = pblockedCovariance[3*2+0];
        covMat(2,1) = pblockedCovariance[3*2+1];
        covMat(2,2) = pblockedCovariance[3*2+2];
        targetCovariances.push_back(covMat);
      }
      gicp.setTargetCovariances(targetCovariances);
    })
  ;

  py::class_<FastVGICP, FastGICP, std::shared_ptr<FastVGICP>>(m, "FastVGICP")
    .def(py::init())
    .def("set_resolution", &FastVGICP::setResolution)
    .def("set_neighbor_search_method", [](FastVGICP& vgicp, const std::string& method) { vgicp.setNeighborSearchMethod(search_method(method)); })
    .def("get_voxel_mean_cov", [](FastVGICP& vgicp){ 
      using meanvec = typename std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>;
      using covvec = typename std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;
      std::pair<meanvec, covvec> data = vgicp.getVoxelMeanCov();
//      return py::make_tuple(data.first, data.second);      
      py::list data0; py::list data1;
      for (const auto& d:data.first) data0.append(d);
      for (const auto& d:data.second) data1.append(d);
      return py::make_tuple(data0, data1);
    })
  ;

#ifdef USE_VGICP_CUDA
  py::class_<FastVGICPCuda, LsqRegistration, std::shared_ptr<FastVGICPCuda>>(m, "FastVGICPCuda")
    .def(py::init())
    .def("set_resolution", &FastVGICPCuda::setResolution)
    .def("set_neighbor_search_method",
      [](FastVGICPCuda& vgicp, const std::string& method, double radius) { vgicp.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_correspondence_randomness", &FastVGICPCuda::setCorrespondenceRandomness)
  ;

  py::class_<NDTCuda, LsqRegistration, std::shared_ptr<NDTCuda>>(m, "NDTCuda")
    .def(py::init())
    .def("set_neighbor_search_method",
      [](NDTCuda& ndt, const std::string& method, double radius) { ndt.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_resolution", &NDTCuda::setResolution)
  ;
#endif

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
