// ----------------------------------------------------------------------------
// -                       Fast Global Registration                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) Intel Corporation 2016
// Qianyi Zhou <Qianyi.Zhou@gmail.com>
// Jaesik Park <syncle@gmail.com>
// Vladlen Koltun <vkoltun@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
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

template<typename vec>
class CApp{
public:    
    int Read(const char* filepath);
    void setRef (int ref);
    void setPoint(vec point);
    vec getPoint ();
    int get_N_neigh();
    void ComputeDist();
    void SearchFLANNTree(flann::Index<flann::L2<float>>* index,
                                vec& input,
                                std::vector<int>& indices,
                                std::vector<float>& dists,
                                int nn);
    void selectNeighbors(int neigh_number);
    void setNormal(vec norm);
    void getError(std::vector<float>* error);
    void getErrorNormalized(std::vector<float>* error);
    vec getNormal();
    std::vector<vec> getNeighborhood();
    float getMoy();
    void getImpact(float error_thresh, int *impact, float *sum);
    void pca(Eigen::Vector3f &dir0, Eigen::Vector3f &dir1, Eigen::Vector3f &dir2);
    float Optimize(float div_fact, float lim_mu, double* mu_init, bool normalize);
    float OptimizePos(int it);
    float OptimizePos1(float div_fact, float lim_mu, double* mu_init);
    float OptimizePos2(float div_fact, float lim_mu, double* mu_init);
    float Refine(float div_fact, float lim_mu, double* mu_init);
    void writeNormal(const char* filepath);
    void writeNeighbors(std::string filepath);
    void writeErrors(std::string filepath);
    void addSymmetricDist();
    void buildTree();
    vec getVecMoy();
    void initNormal();
    void select_normal(int* impact, int impact1, float sum_error, float sum_error1, vec &normal_first2, vec &normal_second2, vec& point_first, vec& point_second);
    void getEdgeDirection(int it);


private:
	// containers
    std::vector<vec> pointcloud_;
    std::vector<vec> neighborhood_;
    std::vector<float> poids_;
    std::vector<vec> dist_;
    int ref_;
    vec pt;
    vec normal;
    float theta;
    float phi;
    std::vector<float> error_;
    std::vector<float> diff_poids_;
    std::vector<float> diff_error_;
    flann::Index<flann::L2<float>>* tree_;
    vec moy_;
    void ComputeWeighs(bool normalize, double mu, bool use_last);   
};
