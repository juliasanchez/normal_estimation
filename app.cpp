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
        moy_ += dist_[i];
    }

    moy_ = moy_/neighborhood_.size();
}

void CApp::initNormal()
{
    normal(0) = sin(theta) * cos(phi);
    normal(1) = sin(theta) * sin(phi);
    normal(2) = cos(theta);
}


void CApp::pca(Eigen::Vector3f &dir0, Eigen::Vector3f &dir1, Eigen::Vector3f &dir2)
{
    Eigen::MatrixXf points (3, neighborhood_.size());

    for (int i = 0; i<neighborhood_.size(); ++i)
    {
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
    dir2 = it->second;
    ++it;
    dir1 = it->second;
    ++it;
    dir0 = it->second;

    float sens = dir2.dot(pt.segment(0,3));
    if(sens>0)
        dir0=-dir0;

    sens = dir1.dot(pt.segment(0,3));
    if(sens>0)
        dir1=-dir1;

    sens = dir2.dot(pt.segment(0,3));
    if(sens>0)
        dir2=-dir2;

    }

void CApp::getEdgeDirection(int it)
{
    initNormal();
    Eigen::Vector3f normal_test, global_min;

    global_min = normal.segment(0,3);

    int N_hist = 10;
    Eigen::MatrixXf hist(N_hist,N_hist);

    normal_test(0) = sqrt(1/(1 + (global_min(0)*global_min(0)/(global_min(1)*global_min(1)) ) ) );
    normal_test(1) = normal_test(0)*(-global_min(0)/global_min(1));
    normal_test(2) = 0;
    normal_test /= normal_test.norm();

    theta = 2*M_PI/it;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate (Eigen::AngleAxisf (theta, global_min));

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
            normal.segment(0,3) = normal_test;
            temp = sum;
        }
    }

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));
}



void CApp::ComputeWeighs(double mu)
{
    std::vector<float> er_angle(neighborhood_.size());
    std::vector<float> er_proj(neighborhood_.size());

    for (int c = 0; c < neighborhood_.size(); c++)
    {
        er_angle[c] = abs(normal.dot(dist_[c]/dist_[c].norm()));
        er_proj[c] = abs(normal.dot(dist_[c]));
    }

    float er_angle_max = *std::max_element(er_angle.begin(), er_angle.end());
    float er_proj_max = *std::max_element(er_proj.begin(), er_proj.end());

    float max_dist = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        if(max_dist<dist_[c].norm())
            max_dist = dist_[c].norm();
    }

    float sigma = coeff_sigma*max_dist;

    for (int c = 0; c < neighborhood_.size(); c++)
    {
        float alpha = ( max_dist-dist_[c].norm() )/max_dist;
        float er_tot = (1-alpha)*er_angle[c]/er_angle_max + (1-alpha) * er_proj[c]/er_proj_max ;
        er_tot = er_tot*er_tot;
        float poids_dist = exp(-(dist_[c].dot(dist_[c]))/(2*sigma*sigma));
        poids_[c] = poids_dist * mu / (er_tot + mu);
    }

    float max_poids = *std::max_element(poids_.begin(), poids_.end());

    for (int c = 0; c < neighborhood_.size(); c++) {
        poids_[c] /= max_poids;
    }

}

void CApp::ComputeWeighs_proj(double mu)
{
    float max_dist = 0;
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        if(max_dist<dist_[c].norm())
            max_dist = dist_[c].norm();
    }

    float er_proj;
    float sigma = coeff_sigma*max_dist;

    for (int c = 0; c < neighborhood_.size(); c++)
    {
        er_proj = normal.dot(dist_[c]);
        float poids_dist = exp(-(dist_[c].dot(dist_[c]))/(2*sigma*sigma));
        poids_[c] = poids_dist * mu / (er_proj*er_proj + mu);
    }

    float max_poids = *std::max_element(poids_.begin(), poids_.end());

    for (int c = 0; c < neighborhood_.size(); c++) {
        poids_[c] /= max_poids;
    }

}


//Once the global minimum of the function is computed we make the minimization WEIGHTED depending on the neighbors error

void CApp::Optimize(float div_fact, float lim_mu, double* mu_init) // normalize decided if minimiser projection ou minimiser angle : projection normalisée
//probleme with angle : points on the same z have different errors : less error for those far from p0 in z direction
//problem with projection : points on the same p0pi direction have different errors : less error for closest points.
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
        mu = std::max(lim_mu,dist[(int)(dist.size()/4)]);
    }

    poids_.resize(neighborhood_.size());
    for (int c = 0; c < neighborhood_.size(); c++) {
        poids_[c] = 0;
    }

    int N = 2;
    int it = std::max(N*(int)(log(mu/(lim_mu) )/log(div_fact)), itr_min);

    const int nvariable = pt.size()-1;	// dimensions of J
    Eigen::MatrixXd JTJ(nvariable, nvariable);
    Eigen::MatrixXd JTr(nvariable, 1);
    Eigen::MatrixXd J(nvariable, 1);
    Eigen::Vector3f normalized;
    double r;

    for( int itr = 1; itr <= it; ++itr)
    {
        if( mu > lim_mu && (itr % N) == 0)
            mu /= div_fact;

        JTJ.setZero();
        JTr.setZero();

        ///actualize normal

        //compute weighs

          ComputeWeighs(mu);

        // minimize function and actualize phi, theta and consequently normal

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            normalized = dist_[c]/dist_[c].norm();
            J(0) = poids_[c] * (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
            J(1) = poids_[c] * (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
            r = poids_[c] * normal.dot(normalized);
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        actuNormal(phi + result(1), theta + result(0));
    }

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    *mu_init = mu;
}


//optional : for points which are in the middle of planes (do not need a weighted minimization) : can optimize alternatively the normal and the point position

void CApp::OptimizePos(int it)
{
    float X = 0;

    int itr = 1;
    float error = 1/epsilon;
    float diff = 1/epsilon;
    float tmp = 0;


    while ( (itr <= it && error>lim_error && diff>lim_diff) || itr<itr_min)
    {
        const int nvariable = 3;
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        double r;
        error = 0;

        //actualize normal

        for (int c = 0; c < neighborhood_.size(); c++) {

            Eigen::Vector3f normalized = dist_[c];
            J(0) = (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
            J(1) = (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
            J(2) = -1;
            r = normal.dot(normalized)-X;
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        actuNormal(phi + result(1), theta + result(0));
        X += result(2);

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            Eigen::Vector3f normalized = dist_[c]-X*normal;
            r = normal.dot(normalized);
            error = error + r*r;
        }

        error = sqrt(error);
        if(itr != 1)
            diff = abs(tmp - error);

        ++itr;

        tmp = error;

    }

    pt = pt + X*normal;
    ComputeDist();

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;
}


//for points on edges : can optimize alternatively the normal and the point position (some points are falsely optimized because of the precision of the normal)

void CApp::OptimizePos1(float div_fact, float lim_mu, double* mu_init)
{
    //we first actualize mu to be in the same situation as in the output of optimize (because we have changed error evaluation)
    float temp_i;
    float tempo = 1/epsilon;

    double mu = *mu_init;

    for (int i = 0; i<dist_.size(); ++i)
    {
        if(tempo>abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2)))
        {
            tempo = abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2));
            temp_i = i;
        }
    }

    mu = std::max(lim_mu,(float)(0.5*pow(normal.dot(dist_[temp_i]),2)));
//    double mu = 10* lim_mu;
    *mu_init = mu;

    int it = std::max(itr_per_mu*(int)(log(mu/(lim_mu) )/log(div_fact)), itr_min);

    const int nvariable = 2;	// two variables : phi and theta of normal
    Eigen::MatrixXd JTJ(nvariable, nvariable);
    Eigen::MatrixXd JTr(nvariable, 1);
    Eigen::MatrixXd J(nvariable, 1);
    double r;
    float sum_poids = 0;
    float moy_proj = 0;

    for( int itr = 1; itr <= it; ++itr)
    {
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

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    *mu_init = mu;
}


//get average point of the neighborhood

Eigen::Vector3f CApp::getVecMoy()
{
    return moy_;
}


//get actual normal

Eigen::Vector3f CApp::getNormal()
{
    return normal;
}


//optional : set the new point (when optimizing position) as reference to be used in further normal computation (better if not needed

void CApp::setRef( int ref)
{
    ref_ = ref;
    pt = pointcloud_->at(ref);
}

void CApp::actuNormal( float phi_new, float theta_new)
{
    phi = phi_new;
    theta = theta_new;
    if(theta == 0)
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


// put the actual error (projection of p_ip_0 on normal) of each point in a vector in parameters

void CApp::getError(std::vector<float>* error)
{
    std::vector<float> err(neighborhood_.size());
    setRef(ref_);
    ComputeDist();
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        err[c] = normal.dot(dist_[c]);
    }
    *error=err;
}


// put the actual error (cos(angle) between p_ip_0 and normal) of each point in a vector

void CApp::getErrorNormalized(std::vector<float>* error)
{
    std::vector<float> err(neighborhood_.size());
    ComputeDist();
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        Eigen::Vector3f normalized = dist_[c]/dist_[c].norm();
        err[c] = normal.dot(normalized);
    }
    *error=err;
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
    std::vector<float> error;

    getError(&error);

    float sum_error = 0;
    float sum_poids = 0;

    for(int j = 0; j<neighborhood_.size(); ++j)
    {
        if(poids_[j]>0.6)
        {
            sum_error += poids_[j] * error[j];
            sum_poids += poids_[j];
            ++imp;
        }
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

void CApp::select_normal(int* impact, int impact1, float sum_error, float sum_error1, Eigen::Vector3f& normal_first2, Eigen::Vector3f& normal_second2, Eigen::Vector3f& point_first, Eigen::Vector3f& point_second)
{
    if(sum_error<sum_error1)
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


std::vector<Eigen::Vector3f> CApp::getNeighborhood()
{
    return neighborhood_;
}
