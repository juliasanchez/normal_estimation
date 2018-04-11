// -------------------------------------------------------------------------------------
//                                Normal estimation
//--------------------------------------------------------------------------------------
//Julia Sanchez, Florent dupont, Florence Denis, Paul Checchin, Laurent Trassoudaine
//Liris (Lyon), Institut Pascal (Clermont Ferrand)
//Région Auvergne Rhone Alpes ARC6
// --------------------------------------------------------------------------------------

#include "app.h"


//build tree to compute neighbors

void CApp::setTree(flann::Index<flann::L2<float>> *t)
{
    tree_ = t;
}

void CApp::setPc(std::vector<Eigen::Vector3f> *pc)
{
    pointcloud_ = pc;
}


// Get neighbors from tree and puts it in neighborhood_

void CApp::selectNeighbors(int neigh_number)
{
    std::vector<float> dis;
    std::vector<int> neigh;

    SearchFLANNTree(tree_, pt, neigh, dis, neigh_number);
    neighborhood_.resize (neigh.size()-1);

    for (int i = 1; i < neigh.size(); ++i)
    {
        Eigen::Vector3f test = pointcloud_->at(neigh[i])-pt;
        if(test.norm() != 0)
            neighborhood_[i-1] = pointcloud_->at(neigh[i]);
        else
        {
            SearchFLANNTree(tree_, pt, neigh, dis, neigh_number+1);
            neigh.erase(neigh.begin()+i -1);
            i = i-1;
        }
    }

    poids_.resize(neighborhood_.size());
}


//get neighborhood size

int CApp::get_N_neigh()
{
    return neighborhood_.size();
}

//Search function in the tree to get the nearest. (internally used in selectNeighbors)

void CApp::SearchFLANNTree(flann::Index<flann::L2<float>>* index,
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


//Once neighborhood is computed, we compute distance of the neighbors from current point and put it in dist_
void CApp::ComputeDist()
{
    dist_.resize (neighborhood_.size());
    for (int i = 0; i < neighborhood_.size(); ++i)
    {
        dist_[i] = neighborhood_[i] - pt;
    }
}


void CApp::pca()
{
    Eigen::Vector3f dir0;
    Eigen::Vector3f dir1;
    Eigen::Vector3f dir2;

    Eigen::MatrixXf points (3, neighborhood_.size());

//    Eigen::Vector3f moy_point = Eigen::Vector3f::Zero();
//    for (int i = 0; i<neighborhood_.size(); ++i)
//        moy_point += neighborhood_[i];
//    moy_point /= neighborhood_.size();

    for (int i = 0; i<neighborhood_.size(); ++i)
    {
//        points(0,i) = neighborhood_[i](0) - moy_point(0);
//        points(1,i) = neighborhood_[i](1) - moy_point(1);
//        points(2,i) = neighborhood_[i](2) - moy_point(2);

        points(0,i) = neighborhood_[i](0) - pt(0);
        points(1,i) = neighborhood_[i](1) - pt(1);
        points(2,i) = neighborhood_[i](2) - pt(2);
    }

    Eigen::Matrix3f covariance = points*points.transpose();

    Eigen::EigenSolver<Eigen::Matrix3f> es(covariance);

    dir0(0) = es.eigenvectors().col(0)[0].real();
    dir0(1) = es.eigenvectors().col(0)[1].real();
    dir0(2) = es.eigenvectors().col(0)[2].real();

    dir1(0) = es.eigenvectors().col(1)[0].real();
    dir1(1) = es.eigenvectors().col(1)[1].real();
    dir1(2) = es.eigenvectors().col(1)[2].real();

    dir2(0) = es.eigenvectors().col(2)[0].real();
    dir2(1) = es.eigenvectors().col(2)[1].real();
    dir2(2) = es.eigenvectors().col(2)[2].real();

    std::multimap<float, Eigen::Vector3f > eigen;
    eigen.insert(std::make_pair(es.eigenvalues()(0).real(), dir0));
    eigen.insert(std::make_pair(es.eigenvalues()(1).real(), dir1));
    eigen.insert(std::make_pair(es.eigenvalues()(2).real(), dir2));

    std::multimap<float, Eigen::Vector3f>::iterator it=eigen.begin();
    setNormal(it->second);
    }

void CApp::getEdgeDirection(int it)
{
    Eigen::Vector3f normal_test;

    int N_hist = 10;
    Eigen::MatrixXf hist(N_hist,N_hist);

    normal_test(0) = sqrt(1/(1 + (normal(0)*normal(0)/(normal(1)*normal(1)) ) ) );
    normal_test(1) = normal_test(0)*(-normal(0)/normal(1));
    normal_test(2) = 0;
    normal_test /= normal_test.norm();

    theta = 2*M_PI/it;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate (Eigen::AngleAxisf (theta, normal));

    float temp = 1/epsilon;

    for(int k = 0; k <= it; k++)
    {
        float x = transform(0,0)*normal_test(0) + transform(0,1)*normal_test(1) + transform(0,2)*normal_test(2) ;
        float y = transform(1,0)*normal_test(0) + transform(1,1)*normal_test(1) + transform(1,2)*normal_test(2) ;
        float z = transform(2,0)*normal_test(0) + transform(2,1)*normal_test(1) + transform(2,2)*normal_test(2) ;

        normal_test(0) = x;
        normal_test(1) = y;
        normal_test(2) = z;

        Eigen::MatrixXf projections(neighborhood_.size(), 3);

        for(int i = 0; i < N_hist; i++)
        {
            for(int j = 0; j < N_hist; j++)
            {
                hist(i,j) = 0;
            }
        }
        normal_test /= normal_test.norm();
        float alpha = acos(normal_test(2)); //angle between normal_test and z angle
        Eigen::Affine3f align_z = Eigen::Affine3f::Identity();
        Eigen::Vector3f axis = {normal_test(1), -normal_test(0), 0};
        axis /= axis.norm();
        align_z.rotate (Eigen::AngleAxisf (alpha, axis));

        for(int c = 0; c < neighborhood_.size(); c++)
        {
            Eigen::Vector3f neigh = neighborhood_[c].segment(0,3);
            neigh = align_z * neigh;
            Eigen::Vector3f projection = {neigh(0), neigh(1), 0};
            projections.row(c) = projection;
        }

        float max_projx = projections.col(0).maxCoeff();
        float min_projx = projections.col(0).minCoeff();
        float max_projy = projections.col(1).maxCoeff();
        float min_projy = projections.col(1).minCoeff();


        for(int c = 0; c < neighborhood_.size(); c++)
        {
            float bin_widthx = (max_projx - min_projx)/((float)N_hist-1);
            float bin_widthy = (max_projy - min_projy)/((float)N_hist-1);

            int a = (projections(c,0)-min_projx)/bin_widthx;
            int b = (projections(c,1)-min_projy)/bin_widthy;
            ++hist(a,b);
        }

        int sum = 0;

        for(int i = 0; i < N_hist; i++)
        {
            for(int j = 0; j < N_hist; j++)
            {
                if(hist(i,j) > 0)
                    ++sum;
            }
        }

        if(temp>sum)
        {
            setNormal(normal_test);
            temp = sum;
        }
    }
}

void CApp::ComputeDistWeighs()
{
    dist_weighs_.resize(neighborhood_.size());
    float max_dist = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        if(max_dist<dist_[c].norm())
            max_dist = dist_[c].norm();
    }

    float sigma = coeff_sigma*max_dist;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        dist_weighs_[c] = exp(-(dist_[c].dot(dist_[c]))/(2*sigma*sigma));
    }
}

void CApp::ComputeWeighs(double mu)
{
//    std::vector<float> er_angle(neighborhood_.size());
//    std::vector<float> er_proj(neighborhood_.size());

//    float max_dist = 0;
//    float er_angle_max = 0;
//    float er_proj_max = 0;
//    for (int c = 0; c < neighborhood_.size(); c++)
//    {
//        er_angle[c] = abs(normal.dot(dist_[c]/dist_[c].norm()));
//        er_proj[c] = abs(normal.dot(dist_[c]));

//        if(er_angle_max<er_angle[c])
//            er_angle_max = er_angle[c];

//        if(er_proj_max<er_proj[c])
//            er_proj_max = er_proj[c] ;
//    }
//        max_dist = *std::max_element(dist_.begin(), dist_.end());

    std::vector<float> er_tot(neighborhood_.size());
    ComputeTotalError(er_tot);

    float max_poids = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
//        float alpha = ( max_dist-dist_[c].norm() )/max_dist;
//        float er_tot = (1-alpha)*er_angle[c]/er_angle_max + (1-alpha) * er_proj[c]/er_proj_max;
//        poids_[c] = dist_weighs_[c] * mu / (er_tot*er_tot + mu);
        poids_[c] = dist_weighs_[c] * mu / (er_tot[c]*er_tot[c] + mu);
        if(max_poids < poids_[c])
            max_poids  = poids_[c];
    }

    for (int c = 0; c < neighborhood_.size(); c++)
    {
        poids_[c] /= max_poids;
    }
}

void CApp::ComputeTotalError(std::vector<float>& er_tot)
{
    std::vector<float> er_angle(neighborhood_.size());
    std::vector<float> er_proj(neighborhood_.size());

    float max_dist = 0;
    float er_angle_max = 0;
    float er_proj_max = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        er_angle[c] = abs(normal.dot(dist_[c]/dist_[c].norm()));
        er_proj[c] = abs(normal.dot(dist_[c]));

        if(er_angle_max<er_angle[c])
            er_angle_max = er_angle[c];

        if(er_proj_max<er_proj[c])
            er_proj_max = er_proj[c] ;
    }

    max_dist = dist_[neighborhood_.size()-1].norm();
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        float alpha = ( max_dist-dist_[c].norm() )/max_dist;
        er_tot[c] = (1-alpha)*er_angle[c]/er_angle_max + (1-alpha) * er_proj[c]/er_proj_max;
    }
}

void CApp::ComputeWeighs_proj(double mu)
{
    float er_proj;
    float max_poids = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        er_proj = normal.dot(dist_[c]);
        poids_[c] = dist_weighs_[c] * mu / (er_proj*er_proj + mu);
        if(max_poids < poids_[c])
            max_poids  = poids_[c];
    }

    for (int c = 0; c < neighborhood_.size(); c++)
        poids_[c] /= max_poids;

}


//Once the global minimum of the function is computed we make the minimization WEIGHTED depending on the neighbors error

void CApp::Optimize(float div_fact, float lim_mu, double* mu_init)
{
    double mu = *mu_init;

    if (mu < epsilon)
    {
        mu = mu_max;
    }
    else
    {
        std::vector<float> dist(dist_.size());
        for (int i = 0; i<dist.size(); ++i)
        {
            Eigen::Vector3f normalized = dist_[i]/dist_[i].norm();
            dist[i] = normal.dot(normalized) * normal.dot(normalized);
        }

        std::sort(dist.begin(), dist.end());
        mu = std::max(lim_mu,(float)dist[(int)(dist.size()/4)]);
//        mu = 5*std::max(lim_mu,dist[(int)(dist.size())]);
    }

    int it = std::max(itr_per_mu*(int)(log(mu/(lim_mu) )/log(div_fact)), itr_min);

    const int nvariable = pt.size()-1;	// dimensions of J
    Eigen::MatrixXd JTJ(nvariable, nvariable);
    Eigen::MatrixXd JTr(nvariable, 1);
    Eigen::MatrixXd J(nvariable, 1);
    Eigen::Vector3f normalized;
    int imp = 0;

    int itr = 0;
//    for( int itr = 1; itr <= it; ++itr)
    while(imp<(int)(0.75*neighborhood_.size()) && itr<it)
    {
        if( mu > lim_mu && (itr % itr_per_mu) == 0)
            mu /= div_fact;

        JTJ.setZero();
        JTr.setZero();

        ///actualize normal

//        if(itr%10 == 0 || itr<10)
//            save_itr(itr);

        //compute weighs
          ComputeWeighs(mu);

        // minimize function and actualize phi, theta and consequently normal

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            normalized = dist_[c]/dist_[c].norm();
            J(0) = poids_[c] * (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
            J(1) = poids_[c] * (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
            JTJ += J * J.transpose();
            JTr += J * poids_[c] * normal.dot(normalized);
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        actuNormal(phi + result(1), theta + result(0));

        imp = 0;
        for (int c = 0; c < neighborhood_.size(); c++)
        {
            if(poids_[c] < 0.1)
            {
                ++imp;
            }
        }

        ++itr;

    }

    *mu_init = mu;
}

void CApp::OptimizePos(int it)
{
    float X = 0;
    const int nvariable = 3;
    Eigen::MatrixXd JTJ(nvariable, nvariable);
    Eigen::MatrixXd JTr(nvariable, 1);
    Eigen::MatrixXd J(nvariable, 1);

    it = std::max(it, itr_min);

    for (int itr = 0; itr <= it; ++itr)
    {
        JTJ.setZero();
        JTr.setZero();

        //actualize normal

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            J(0) = (dist_[c](0) * cos(theta) * cos(phi)    + dist_[c](1) * cos(theta) * sin(phi) + dist_[c](2) * (-sin(theta)));
            J(1) = (dist_[c](0) * sin(theta) * (-sin(phi)) + dist_[c](1) * sin(theta) * cos(phi));
            J(2) = -1;
            JTJ += J * J.transpose();
            JTr += J * (normal.dot(dist_[c])-X);
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        actuNormal(phi + result(1), theta + result(0));
        X += result(2);
    }

    pt = pt + X*normal;

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;
}


//for points on edges : can optimize alternatively the normal and the point position

void CApp::OptimizePos1(float div_fact, float lim_mu, double* mu_init, bool convert_mu)
{
     double mu = *mu_init;
//    system("rm *.csv");

    if(convert_mu) //we first actualize mu to be in the same situation as in the output of optimize (because we have changed error evaluation)
    {
        std::vector<float> er_tot(neighborhood_.size());
        ComputeTotalError(er_tot);
        float new_mu = 0;
        float J_mu;
        float J_muTJ_mu = 0;
        float  J_muTr = 0;
        float r;
        float error = 0;
        float tmp_error = 1/epsilon;
        std::vector<float> old_weigh(neighborhood_.size());
        std::vector<float> er_proj2(neighborhood_.size());

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            old_weigh[c] = mu/(mu+er_tot[c]*er_tot[c]);
            er_proj2[c] = normal.dot(dist_[c])*normal.dot(dist_[c]);
        }

        int n = 0;
        while (tmp_error-error>epsilon || n<2)
        {
            J_muTJ_mu = 0;
            J_muTr = 0;

            for (int c = 0; c < neighborhood_.size(); c++)
            {
                J_mu = er_proj2[c]/pow(new_mu + er_proj2[c],2);
                r = new_mu/(new_mu+er_proj2[c]) - old_weigh[c];
                J_muTJ_mu += J_mu * J_mu;
                J_muTr += J_mu * r;
            }

            new_mu += -J_muTr/J_muTJ_mu;

            tmp_error = error;
            error = 0;
            for (int c = 0; c < neighborhood_.size(); c++)
            {
                r = new_mu / (new_mu + er_proj2[c]) - old_weigh[c];
                error += r*r;
            }
            ++n;
        }

        mu = std::max(lim_mu, new_mu);

        *mu_init = mu;
    }

    int it = std::max(itr_per_mu*(int)(log(mu/(lim_mu) )/log(div_fact)), itr_min);

    const int nvariable = 2;	// two variables : phi and theta of normal
    Eigen::MatrixXd JTJ(nvariable, nvariable);
    Eigen::MatrixXd JTr(nvariable, 1);
    Eigen::MatrixXd J(nvariable, 1);
    float sum_poids = 0;
    float moy_proj = 0;
    float r;

    for( int itr = 1; itr <= it; ++itr)
    {
//        if(itr%10 == 0 || itr<10)
//            save_itr(itr);

        if( mu > lim_mu && (itr % itr_per_mu) == 0)
            mu /= div_fact;

        JTJ.setZero();
        JTr.setZero();

        //actualize position of the point

        sum_poids = 0;
        moy_proj = 0;

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            if(poids_[c] > thresh_weigh)
            {
                r = poids_[c]*normal.dot(dist_[c]);
                sum_poids += poids_[c];
                moy_proj += r;
            }
//            if(pow(normal.dot(dist_[c]),2)<0.25*mu)
//            {
//                r = poids_[c]*normal.dot(dist_[c]);
//                sum_poids += poids_[c];
//                moy_proj += r;
//            }
        }
        moy_proj = moy_proj/sum_poids;
        pt = pt + moy_proj*normal;

        ComputeDist();

        //Compute weighs

        ComputeWeighs_proj(mu);

        //actualize normal

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            J(0) = poids_[c] * (dist_[c](0) * cos(theta) * cos(phi)    + dist_[c](1) * cos(theta) * sin(phi) + dist_[c](2) * (-sin(theta)));
            J(1) = poids_[c] * (dist_[c](0) * sin(theta) * (-sin(phi)) + dist_[c](1) * sin(theta) * cos(phi));
            r = poids_[c] * normal.dot(dist_[c]);
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        actuNormal(phi + result(1), theta + result(0));
    }

     orient();

    *mu_init = mu;
}

//orient normal to the exterior of the edge

void CApp::orient()
{
    float moy_err = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        moy_err += normal.dot(dist_[c]);
    }

    if(moy_err>0.005)
        normal = -normal;
}

//get actual normal

Eigen::Vector3f CApp::getNormal()
{
    return normal;
}


//optional : set the new point (when optimizing position) as reference to be used in further normal computation

void CApp::setRef( int ref)
{
    ref_ = ref;
    pt = pointcloud_->at(ref);
}

void CApp::actuNormal( float phi_new, float theta_new)
{
    phi = phi_new;
    theta = theta_new;
    if(theta < epsilon)
        theta = theta_min;

    normal(0) = sin(theta) * cos(phi);
    normal(1) = sin(theta) * sin(phi);
    normal(2) = cos(theta);
}


//get actual point

Eigen::Vector3f CApp::getPoint()
{
    return pt;
}

//compute average projection of p_ip_0 on normal

float CApp::getMoy()
{
    ComputeDist();
    Eigen::Vector3f pt_moy = Eigen::Vector3f::Zero();
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        pt_moy += neighborhood_[c];
    }

    pt_moy /= neighborhood_.size();

    float proj_moy = 0;

    for (int c = 0; c < neighborhood_.size(); c++)
    {
        proj_moy += pow(normal.dot(neighborhood_[c]-pt_moy),2);
    }

    proj_moy /= neighborhood_.size();
    proj_moy = sqrt(proj_moy);

     return proj_moy;
}

void CApp::getImpact(int *impact, float *sum)
{
    float imp = 0;

    float sum_error = 0;
    float sum_poids = 0;

    setRef(ref_);
    ComputeDist();

    for(int c = 0; c<neighborhood_.size(); ++c)
    {
        if(poids_[c]>impacter_weigh)
        {
//            sum_error += poids_[c] * normal.dot(dist_[c]);
//            sum_poids += poids_[c];
            sum_error += normal.dot(dist_[c]);
            ++sum_poids;
            ++imp;
        }
//        if(pow(normal.dot(dist_[c]),2)<mu)
//        {
//            sum_error += poids_[c] * normal.dot(dist_[c]);
//            sum_poids += poids_[c];
////            sum_error += normal.dot(dist_[c]);
////            ++sum_poids;
//            ++imp;
//        }
    }

    *impact = imp;

    if(sum_poids>0)
    {
        *sum = sum_error/sum_poids;
    }
}



//writeNormal in file

void CApp::writeNormal(const char* filepath)
{
    std::ofstream fout(filepath, std::ofstream::app);

    if (fout.is_open())
    {
        for (int i = 0; i<pt.size(); ++i)
        {
            fout<<pt(i)<<",";
//            fout<<pointcloud_[ref_](i)<<",";
        }
        for (int i = 0; i<normal.size(); ++i)
        {
            fout<<normal(i)<<",";
        }
        fout<<"\n";
        fout.close();
    }
}


//write neighbors in file

void CApp::writeNeighbors(std::string filepath)
{
    std::ofstream fout(filepath, std::ofstream::app);

    if (fout.is_open())
    {
        for (int i = 0; i<neighborhood_.size(); ++i)
        {
            for (int j = 0; j<3; ++j)
            {
                fout<<neighborhood_[i](j)<<",";
            }
            fout << poids_[i];
            fout<<"\n";
        }
        fout.close();
    }
}

//write errors in file

void CApp::writeErrors(std::string filepath)
{
    std::ofstream fout(filepath, std::ofstream::app);

    if (fout.is_open())
    {
        for (int i = 0; i<neighborhood_.size(); ++i)
        {
            for (int j = 0; j<3; ++j)
            {
                fout<<neighborhood_[i](j)<<",";
            }
            fout << error_[i];
            fout<<"\n";
        }
        fout.close();
    }
}

//set Normal
void CApp::setNormal(Eigen::Vector3f norm)
{
    normal = norm;
    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));
    if(theta == 0)
        theta = theta_min;
}


//set point as reference

void CApp::setPoint(Eigen::Vector3f point)
{
    pt = point;
}

void CApp::select_normal(int impact, int impact1, float sum_error, float sum_error1, Eigen::Vector3f& normal_first2, Eigen::Vector3f& normal_second2, Eigen::Vector3f& point_first, Eigen::Vector3f& point_second)
{
    if( sum_error<sum_error1 && impact>(int)(lim_impacters*(float)neighborhood_.size()) /*&& abs(sum_error)<0.01*/ ) // la limite abs(sum_error1) est dégueux A CHANGER
    {
        normal = normal_first2;
        pt = point_first;
    }
    else if(impact1>(int)(lim_impacters*(float)neighborhood_.size()) /*&& abs(sum_error1)<0.01*/)  // la limite abs(sum_error1) est dégueux A CHANGER
    {
        normal = normal_second2;
        pt = point_second;
    }
    else if(impact>(int)(lim_impacters*(float)neighborhood_.size()) )
    {
        normal = normal_first2;
        pt = point_first;
    }
    else if( sum_error<sum_error1 )
    {
        normal = normal_first2;
        pt = point_first;
    }
    else
    {
        normal = normal_second2;
        pt = point_second;
    }

}


void CApp::save_itr(int itr)
{
    std::ofstream fneigh;
    std::stringstream stm;
    stm.str("");
    stm<<"neighbors"<<itr<<".csv";
    fneigh.open(stm.str(), std::ofstream::trunc);
    if (fneigh.is_open())
    {
        fneigh<<"x,y,z,weight\n";
        for(int j = 0; j<neighborhood_.size(); ++j)
            fneigh<<neighborhood_[j](0)<<","<<neighborhood_[j](1)<<","<<neighborhood_[j](2)<<","<<poids_[j]<<"\n";
    }
    fneigh.close();

    std::ofstream fnormal;
    stm.str("");
    stm<<"normal"<<itr<<".csv";
    fnormal.open(stm.str(), std::ofstream::trunc);
    if (fnormal.is_open())
    {
        fnormal<<"x,y,z,nx,ny,nz\n";
        fnormal<<pt(0)<<","<<pt(1)<<","<<pt(2)<<","<<normal(0)<<","<<normal(1)<<","<<normal(2)<<"\n";
    }
    fnormal.close();
}



std::vector<Eigen::Vector3f> CApp::getNeighborhood()
{
    return neighborhood_;
}
