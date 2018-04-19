#include <vector>
#include <flann/flann.hpp>

#include <fstream>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>
#include "/usr/local/include/Eigen/Core"
#include "/usr/local/include/Eigen/Dense"
#include "/usr/local/include/Eigen/Eigen"
#include "/usr/local/include/Eigen/Eigenvalues"

using namespace std;

//Important constants with real effect on result
const float thresh_weigh = 0.6;     // weigh threshold value for neighbors when moving point with mean projection of neighbors
const float impacter_weigh = 0.6;   // weigh threshold for computing error and evaluate/select normals
const float lim_impacters = 0.1;    // limit % of points of the neighborhood which must have an impact on the normal computation
const float lim_mu = 0.1;           // mu threshold value when optimizing without moving points positions decrease when no noise
const float div_option2 = 6.0;     // initial mu when starting the optimization for selection 2 (mu_max/div_option2)
                                    // decrease if too many outliers/ increase if too many wrong selection
const float coeff_sigma = 2;        // sigma for distance weigh = coeff_sigma*farthestePoint increase if the surface were the point lays is small compared to neighborhood

//Usual constants DO NOT MODIFY
const float epsilon = 1e-10;        // small value for initializations of comparisons
const float lim_error = 1e-5;       // to stop optimize when not moving
const float lim_diff = 1e-7;        // to stop optimize when not moving
const float theta_min = 1e-10;      // because theta can not be 0 for phi to make sense

//Detail constants
const int itr_min = 50;             // minimum number of iterations to perform
const float mu_max = 3.0;           // initial mu when starting optimize for the first time
const float init2_accuracy = 50;    // number of vectors tested to get edge direction (may be decreased to gain time)
const float N_hist = 10;            // number of bins used to find the 2nd initialization
const int itr_per_mu = 1;           // number of iterations to perform for each mu
const float itr_opti_pos_plan = 30; // number of iterations to optimize position of point in plane configuration
const float noise_min = 0.001;      // minimum noise to make it not infinite
const float noise_max = 0.01;       // maximum noise to force iterations when very noisy
const float likelihood_threshold = 0.95; // value for evaluation/comparison between vectors


class CApp{
public:
    CApp()
    {
        std::cout<<"add one pointcloud + its kdtree + the reference point you want to compute the normal on"<<std::endl<<std::endl;
        normalFirst0_ = NULL;
        normalFirst1_ = NULL;
        normalFirst2_ = NULL;
        normalSecond0_ = NULL;
        normalSecond1_ = NULL;
        normalSecond2_ = NULL;
        pointFirst_ = NULL;
        pointSecond_ = NULL;
    };

    CApp(std::vector<Eigen::Vector3f>*pc, flann::Index<flann::L2<float>>* tree, float noise)
    {
        setPc(pc);
        setTree(tree);
        noise_ = noise;
        normalFirst0_ = NULL;
        normalFirst1_ = NULL;
        normalFirst2_ = NULL;
        normalSecond0_ = NULL;
        normalSecond1_ = NULL;
        normalSecond2_ = NULL;
        pointFirst_ = NULL;
        pointSecond_ = NULL;
    };

    CApp(std::vector<Eigen::Vector3f>*pc, flann::Index<flann::L2<float>>* tree, int ref, float noise)
    {
        setPc(pc);
        setTree(tree);
        setRef(ref);
        noise_ = noise;
        normalFirst0_ = NULL;
        normalFirst1_ = NULL;
        normalFirst2_ = NULL;
        normalSecond0_ = NULL;
        normalSecond1_ = NULL;
        normalSecond2_ = NULL;
        pointFirst_ = NULL;
        pointSecond_ = NULL;
    };

    CApp(float divFact, float limMu, float limMuPos, std::vector<Eigen::Vector3f>*pc, flann::Index<flann::L2<float>>* tree, int ref, float noise)
    {
        divFact_ = divFact;
        limMu_ = limMu;
        limMuPos_ = limMuPos;
        noise_ = noise;
        setPc(pc);
        setTree(tree);
        setRef(ref);
        normalFirst0_ = NULL;
        normalFirst1_ = NULL;
        normalFirst2_ = NULL;
        normalSecond0_ = NULL;
        normalSecond1_ = NULL;
        normalSecond2_ = NULL;
        pointFirst_ = NULL;
        pointSecond_ = NULL;

    };

    ~CApp()
    {
        if(normalFirst0_!= NULL)
            delete normalFirst0_;
        if(normalFirst1_!= NULL)
            delete normalFirst1_;
        if(normalFirst2_!= NULL)
            delete normalFirst2_;
        if(normalSecond0_!= NULL)
            delete normalSecond0_;
        if(normalSecond1_!= NULL)
            delete normalSecond1_;
        if(normalSecond2_!= NULL)
            delete normalSecond2_;
        if(pointFirst_!= NULL)
            delete pointFirst_;
        if(pointSecond_!= NULL)
            delete pointSecond_;
    };

    void setTree(flann::Index<flann::L2<float>> *t);
    void setPc(std::vector<Eigen::Vector3f> *pc);
    void setRef (int ref);
    void setNormal(Eigen::Vector3f norm);
    void setPoint(Eigen::Vector3f point);
    void setLimMu ( float limMu){limMu_ = limMu;};
    void setLimMuPos ( float limMuPos){limMuPos_ = limMuPos;};
    void setDivFact ( float divFact){divFact_= divFact;};
    void setNoise ( float noise){noise_= noise;};
    void setParams(float divFact, float limMu, float limMuPos)
    {
        limMu_ = limMu;
        limMuPos_ = limMuPos;
        divFact_= divFact;
    };
    Eigen::Vector3f getNormal();
    Eigen::Vector3f getPoint ();
    std::vector<Eigen::Vector3f> getNeighborhood();
    int get_N_neigh();

    void reinitPoint();
    void ComputeDist();
    void SearchFLANNTree(flann::Index<flann::L2<float>>* index,
                                Eigen::Vector3f& input,
                                std::vector<int>& indices,
                                std::vector<float>& dists,
                                int nn);
    void selectNeighbors(int neigh_number);

    void init1();
    void reinitFirst0();
    void setFirst2();
    void init2();
    void Optimize(bool first);
    void OptimizePos(int it);
    void OptimizePos1(bool first);
    void getEdgeDirection(int it);
    void evaluate(int *impact, float *moyError);
    void select_normal();
    void addSymmetricDist();
    bool isOnEdge();
    bool isSecondOption();



    bool isNan();
    Eigen::Vector3f finalNormal_;
    Eigen::Vector3f finalPos_;


private:
	// containers
    float mu_;
    float noise_;
    float limMu_;
    float limMuPos_;
    float divFact_;
    std::vector<Eigen::Vector3f> *pointcloud_;
    flann::Index<flann::L2<float>>* tree_;
    std::vector<Eigen::Vector3f> neighborhood_;
    std::vector<Eigen::Vector3f> dist_;
    std::vector<float> dist_weighs_;
    int ref_;
    Eigen::Vector3f pt;
    Eigen::Vector3f ptRef_;
    Eigen::Vector3f normal;
    Eigen::Vector3f* normalFirst0_;
    Eigen::Vector3f* normalFirst1_;
    Eigen::Vector3f* normalFirst2_;
    Eigen::Vector3f* normalSecond0_;
    Eigen::Vector3f* normalSecond1_;
    Eigen::Vector3f* normalSecond2_;
    Eigen::Vector3f* pointFirst_;
    Eigen::Vector3f* pointSecond_;
    float theta;
    float phi;
    std::vector<float> poids_;
    std::vector<float> error_;
    std::vector<float> diff_error_;
    void ComputeDistWeighs();
    void ComputeWeighs();
    void ComputeWeighs_proj();
    void actuNormal( float phi_new, float theta_new);
    void ComputeTotalError(std::vector<float>& er_tot);
    void actualizeMu();
    void orient();
    void save_itr(int itr);
    int impactFirst_;
    int impactSecond_;
    float moyErrorFirst_;
    float moyErrorSecond_;

};
