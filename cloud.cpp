// -------------------------------------------------------------------------------------
//        Iterative weighted PCA for robust and edge-aware normal vector estimation
//--------------------------------------------------------------------------------------
// Julia Sanchez, Florence Denis, David Coeurjolly, Florent dupont, Laurent Trassoudaine, Paul Checchin
// Liris (Lyon), Institut Pascal (Clermont Ferrand)
// Région Auvergne Rhône Alpes ARC6
// Private use for reviewers only
// --------------------------------------------------------------------------------------


#include "cloud.h"

// Reads file and puts it into pointcloud_
//take .pcd .ply .csv .xyz with or without header
int cloud::Read(std::string filepath)
{
    std::cout<<"reading file : "<<filepath<<std::endl<<std::endl;
    std::ifstream fin(filepath);
    int n_points = 0;
    if (fin.is_open())
    {
        //skip header
        string test;
        bool header = true;
        int separators;
        int lines_header = -1;
        while(header)
        {
          separators = 0;
          header = false;
          ++lines_header;
          std::getline ( fin, test, '\n' );
          for(int i = 0; i<test.size(); ++i)
          {
              if(test[i]==',' || test[i]==' ' || test[i]=='\t')
              {
                  ++separators;
                  if(i<test.size())
                  {
                      while( (test[i]==',' || test[i]==' ' || test[i]=='\t'))
                        ++i;
                      --i;
                  }
              }
              else if ( ( test[i]<'0' || test[i]>'9' ) && test[i]!='-' && test[i]!='.' && test[i]!='e')
              {
                  header = true;
                  break;
              }
          }
          if(separators<2)
            header = true;
        }
        //count lines/ points
        n_points = 1;
        while(std::getline ( fin, test, '\n' ))
            ++n_points;
        fin.close();
        fin.open(filepath);
        for(int i = 0; i < lines_header; ++i)
            std::getline ( fin, test, '\n' );

        pointcloud_ = new Eigen::Matrix<float, Eigen::Dynamic, 3>(n_points, 3);

        //load points (skip " ", ",", "\t") x y z must be the first elements of each line
        n_points = 0;
        while (std::getline ( fin, test, '\n' ))
        {
            Eigen::Vector3f pt;
            std::stringstream stm;
            stm.str("");
            int n = 0;
            for(int i = 0; i<test.size(); ++i)
            {
                if(test[i]==',' || test[i]==' ' || test[i]=='\t' || i == test.size()-1)
                {
                    pt(n) = stof(stm.str());
                    ++n;
                    stm.str("");
                    if(i<test.size())
                    {
                        while( (test[i]==',' || test[i]==' ' || test[i]=='\t'))
                          ++i;
                        --i;
                    }
                }
                else
                  stm<<test[i];
                if(n == 3)
                  break;
            }
            pointcloud_->row(n_points) = pt;
            ++n_points;
        }
        fin.close();
    }
    else
       std::cerr<<"did not find file"<<std::endl<<std::endl;

    std::cout<<"number of points  : "<<n_points<<std::endl<<std::endl;
    return n_points;
}

//builds tree to extract neighbors
void cloud::buildTree()
{
    int dim = pointcloud_->cols();
    int pc_size = pointcloud_->rows();

    std::vector<float> pc(pc_size * dim);
    flann::Matrix<float> flann_mat(&pc[0], pc_size, dim);

    int n = 0;

    for (int i =0; i < pc_size; ++i)
    {
        for (int j =0; j < dim; ++j)
        {
            pc[n] = (*pointcloud_)(i,j);
            ++n;
        }
    }

    tree_ = new flann::Index<flann::L2<float>>(flann_mat, flann::KDTreeSingleIndexParams(15));
    tree_->buildIndex();
}

//computes resolution as mean distance between nearest neighbors
float cloud::getResolution ()
{
  float res = 0.0;

  for (int i = 0; i<pointcloud_->rows(); ++i)
      res += getNearestNeighborDistance(pointcloud_->row(i));

  res /= pointcloud_->rows();
  return res;
}

void cloud::rescale ()
{
    float resolution = getResolution();
    std::cout<<"current resolution : "<<resolution<<std::endl<<std::endl;
    *pointcloud_ *= (float)(0.001)/resolution;
}

// Gets first neighbor distance
float cloud::getNearestNeighborDistance(Eigen::Vector3f pt)
{
    std::vector<float> dis;
    std::vector<int> neigh;

    SearchFLANNTree(tree_, pt, neigh, dis, 3);
    if(dis[1] != 0)
        return sqrt(dis[1]);
    else
        return sqrt(dis[2]);
}


//Search function in the tree to get the nearest neighbors
void cloud::SearchFLANNTree(flann::Index<flann::L2<float>>* index,
                            Eigen::Vector3f& input,
                            std::vector<int>& indices,
                            std::vector<float>& dists,
                            int nn)
{
    int dim = input.size();

    std::vector<float> query;
    query.resize(dim);
    for (int i = 0; i < dim; i++)
        query[i] = input(i);
    flann::Matrix<float> query_mat(&query[0], 1, dim);

    indices.resize(nn);
    dists.resize(nn);
    flann::Matrix<int> indices_mat(&indices[0], 1, nn);
    flann::Matrix<float> dists_mat(&dists[0], 1, nn);

    index->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}
