#include <vector>
#include <flann/flann.hpp>

#include <fstream>
#include <math.h>
#include <algorithm>
#include <utility>
#include <vector>
#include "/usr/local/include/Eigen/Core"
#include "/usr/local/include/Eigen/Dense"
#include "/usr/local/include/Eigen/Eigen"
#include "/usr/local/include/Eigen/Eigenvalues"

//using namespace Eigen;
using namespace std;

class cloud{
public:
    cloud();
    int Read(const char* filepath);
    void buildTree();
    std::vector<Eigen::Vector3f>* getPC();
    flann::Index<flann::L2<float>>* getTree();

private:
	// containers
    std::vector<Eigen::Vector3f> pointcloud_;
    flann::Index<flann::L2<float>>* tree_;
};
