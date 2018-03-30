#include "cloud.h"

cloud::cloud()
{

}

// reads file and puts it into pointcloud_

int cloud::Read(const char* filepath)
{
    std::cout<<"reading file"<<std::endl<<std::endl;
    std::ifstream fin(filepath);
    vector<Eigen::Vector3f> res;
    int n = 0;
    if (fin.is_open())
    {
        string test;
        std::getline ( fin, test, '\n' );
        while (std::getline ( fin, test, ',' ))
        {
            Eigen::Vector3f pt;
            pt(0) = stof(test);
            std::getline ( fin, test, ',' );
            pt(1) = stof(test);
            std::getline ( fin, test, '\n' );
            pt(2) = stof(test);

            res.push_back(pt);
            ++n;
        }
        fin.close();
        pointcloud_ = res;
    }
    else
    {
       std::cout<<"did not find file : "<<filepath<<std::endl<<std::endl;
    }
    return n;
}



//build tree to compute neighbors

void cloud::buildTree()
{
    int dim = pointcloud_[0].size();
    int pc_size = pointcloud_.size();

    std::vector<float> pc(pc_size * dim);
    flann::Matrix<float> flann_mat(&pc[0], pc_size, dim);

    int n = 0;

    for (int i =0; i < pc_size; ++i)
    {
        for (int j =0; j < dim; ++j)
        {
            pc[n] = pointcloud_[i](j);
            ++n;
        }
    }

    tree_ = new flann::Index<flann::L2<float>>(flann_mat, flann::KDTreeSingleIndexParams(15));
    tree_->buildIndex();

}


std::vector<Eigen::Vector3f>* cloud::getPC()
{
    return &pointcloud_;
}

flann::Index<flann::L2<float>>* cloud::getTree()
{
    return tree_;
}
