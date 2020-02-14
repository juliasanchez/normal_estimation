// -------------------------------------------------------------------------------------
//        Iterative weighted PCA for robust and edge-aware normal vector estimation
//--------------------------------------------------------------------------------------
// Julia Sanchez, Florence Denis, David Coeurjolly, Florent dupont, Laurent Trassoudaine, Paul Checchin
// Liris (Lyon), Institut Pascal (Clermont Ferrand)
// Région Auvergne Rhône Alpes ARC6
// Private use for reviewers only
// --------------------------------------------------------------------------------------


#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <utility>

#include "Core"
#include "Dense"
#include "Eigenvalues"
#include "flann/flann.hpp"

using namespace std;

class cloud{
public:
    cloud(){}
    int Read(std::string filepath);
    void buildTree();
    Eigen::Matrix<float, Eigen::Dynamic, 3> * getPC(){return pointcloud_;}
    flann::Index<flann::L2<float>>* getTree(){return tree_;}
    float getResolution ();
    void rescale();

private:
    Eigen::Matrix<float, Eigen::Dynamic, 3> *pointcloud_;
    flann::Index<flann::L2<float>>* tree_;
    float getNearestNeighborDistance(Eigen::Vector3f pt);
    void SearchFLANNTree(flann::Index<flann::L2<float>>* index,
                                Eigen::Vector3f& input,
                                std::vector<int>& indices,
                                std::vector<float>& dists,
                                int nn);
};
