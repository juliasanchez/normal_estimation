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

void CApp::setPc(Eigen::Matrix<float, Eigen::Dynamic, 3> *pc)
{
    pointcloud_ = pc;
}


// Get neighbors from tree and puts it in neighborhood_

void CApp::selectNeighbors(int neigh_number)
{
    std::vector<float> dis;
    std::vector<int> neigh;

    SearchFLANNTree(tree_, pt, neigh, dis, neigh_number);
    neighborhood_.resize (neigh.size()-1,3);

    for (int i = 1; i < neigh.size(); ++i)
    {
        if(  ( (pointcloud_->row(neigh[i])).transpose()-pt ).norm() != 0)
            neighborhood_.row(i-1) = pointcloud_->row(neigh[i]);
        else
        {
            SearchFLANNTree(tree_, pt, neigh, dis, neigh_number+1);
            neigh.erase(neigh.begin()+i -1);
            i = i-1;
        }
    }

    weighs_.resize(neighborhood_.rows());

    ComputeDist();
    ComputeDistWeighs();
}


//get neighborhood size

int CApp::get_N_neigh()
{
    return neighborhood_.rows();
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
    dist_.resize (neighborhood_.rows(), 3);
    dist_ = neighborhood_ - Eigen::VectorXf::Ones(neighborhood_.rows()) * pt.transpose();
}

Eigen::Vector3f CApp::getThirdEigenVector(Eigen::Matrix3f& C)
{
    Eigen::Vector3f dir0;
    Eigen::Vector3f dir1;
    Eigen::Vector3f dir2;

    Eigen::EigenSolver<Eigen::Matrix3f> es(C);

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
    return it->second;
}

void CApp::init1()
{
    Eigen::MatrixXf points (neighborhood_.rows(), 3);

    points = neighborhood_ - Eigen::VectorXf::Ones(neighborhood_.rows())*pt.transpose();

    Eigen::Matrix3f covariance = points.transpose()*points;

    setNormal(getThirdEigenVector(covariance));

    normalFirst0_ = new Eigen::Vector3f();
    *normalFirst0_ = normal;
}

void CApp::reinitFirst0()
{
    normal = *normalFirst0_;
}

void CApp::setFirst2()
{
    finalNormal_ = *normalFirst2_;
}

void CApp::init2()
{
    setNormal(*normalFirst2_);
    getEdgeDirection(init2_accuracy);
    normal = normal.cross(*normalFirst2_);
    setNormal(normal/normal.norm());

    normalSecond0_ = new Eigen::Vector3f();
    *normalSecond0_ = normal;
}

bool CApp::isOnEdge()
{
    ComputeDist();
    Eigen::Vector3f pt_moy = Eigen::Vector3f::Zero();
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        pt_moy += neighborhood_.row(c);
    }

    pt_moy /= neighborhood_.rows();

    float proj_moy = 0;

    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        proj_moy += pow(normal.dot((neighborhood_.row(c)).transpose()-pt_moy),2);
    }

    proj_moy /= neighborhood_.rows();
    proj_moy = sqrt(proj_moy);

    return ( proj_moy>noise_ );
}

void CApp::getEdgeDirection(int it)
{
    Eigen::Vector3f normal_test;

    int N_hist = 10;
    Eigen::MatrixXf hist(N_hist,N_hist);

    if(abs(normal(1))<epsilon)
        normal(1) = epsilon;
    normal_test(0) = sqrt(1/(1 + (normal(0)*normal(0)/(normal(1)*normal(1)) ) ) );
    normal_test(1) = normal_test(0)*(-normal(0)/normal(1));
    normal_test(2) = 0;
    normal_test /= normal_test.norm();

    theta = M_PI/it;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate (Eigen::AngleAxisf (theta, normal));

    float temp = 1/epsilon;

    for(int k = 0; k <= it; k++)
    {
        normal_test = transform * normal_test;

        Eigen::MatrixXf projections(neighborhood_.rows(), 3);

        hist.setZero();

        normal_test /= normal_test.norm();
        float alpha = acos(normal_test(2)); //angle between normal_test and z angle
        Eigen::Affine3f align_z = Eigen::Affine3f::Identity();
        Eigen::Vector3f axis = {normal_test(1), -normal_test(0), 0};
        axis /= axis.norm();
        align_z.rotate (Eigen::AngleAxisf (alpha, axis));

        for(int c = 0; c < neighborhood_.rows(); c++)
        {
            Eigen::Vector3f neigh = neighborhood_.row(c).transpose();
            neigh = align_z * neigh;
            Eigen::Vector3f projection = {neigh(0), neigh(1), 0};
            projections.row(c) = projection;
        }

        float max_projx = projections.col(0).maxCoeff();
        float min_projx = projections.col(0).minCoeff();
        float max_projy = projections.col(1).maxCoeff();
        float min_projy = projections.col(1).minCoeff();


        for(int c = 0; c < neighborhood_.rows(); c++)
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
    dist_weighs_.resize(neighborhood_.rows());
    float max_dist = 0;
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        if(max_dist<dist_.row(c).norm())
            max_dist = dist_.row(c).norm();
    }

    float sigma = coeff_sigma*max_dist;
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        dist_weighs_(c) = exp(-(dist_.row(c).dot(dist_.row(c)))/(2*sigma*sigma));
    }
}

void CApp::ComputeWeighs()
{
    std::vector<float> er_tot(neighborhood_.rows());
    ComputeTotalError(er_tot);

    float max_poids = 0;
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        weighs_(c) = dist_weighs_(c) * mu_ / (er_tot[c]*er_tot[c] + mu_);
        if(max_poids < weighs_(c))
            max_poids  = weighs_(c);
    }

    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        weighs_(c) /= max_poids;
        weighs_(c) = weighs_(c) * weighs_(c);
    }


//    std::vector<float> weighs_cpy = weighs_;
//    std::sort(weighs_cpy.begin(), weighs_cpy.end());
//    max_poids = weighs_cpy[(int)(0.9*weighs_.size())];

//    for (int c = 0; c < neighborhood_.size(); c++)
//    {
//        if(weighs_[c] < max_poids )
//            weighs_[c] /= max_poids;
//        else
//            weighs_[c] = 1.0;
//    }

}

void CApp::ComputeTotalError(std::vector<float>& er_tot)
{
    std::vector<float> er_angle(neighborhood_.rows());
    std::vector<float> er_proj(neighborhood_.rows());

    float max_dist = 0;
    float er_angle_max = 0;
    float er_proj_max = 0;
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        er_angle[c] = abs(normal.dot(dist_.row(c)/dist_.row(c).norm()));
        er_proj[c] = abs(normal.dot(dist_.row(c)));

        if(er_angle_max<er_angle[c])
            er_angle_max = er_angle[c];

        if(er_proj_max<er_proj[c])
            er_proj_max = er_proj[c] ;
    }

    max_dist = dist_.row(neighborhood_.rows()-1).norm();
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        float alpha = ( max_dist-dist_.row(c).norm() )/max_dist;
        er_tot[c] = (1-alpha)*er_angle[c]/er_angle_max + alpha * er_proj[c]/er_proj_max;
    }
}

void CApp::ComputeWeighs_proj()
{
    float er_proj;
    float max_poids = 0;
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        er_proj = normal.dot(dist_.row(c));
        weighs_(c) = dist_weighs_(c) * mu_ / (er_proj*er_proj + mu_);
        if(max_poids < weighs_(c))
            max_poids  = weighs_(c);
    }

    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        weighs_(c) /= max_poids;
        weighs_(c) = weighs_(c) * weighs_(c);
    }


//    std::vector<float> weighs_cpy = weighs_;
//    std::sort(weighs_cpy.begin(), weighs_cpy.end());
//    max_poids = weighs_cpy[(int)(0.9*weighs_.size())];

//    for (int c = 0; c < neighborhood_.rows(); c++)
//    {
//        if(weighs_[c] < max_poids )
//            weighs_[c] /= max_poids;
//        else
//            weighs_[c] = 1.0;
//    }


}


//Once the global minimum of the function is computed we make the minimization WEIGHTED depending on the neighbors error

void CApp::Optimize(bool first)
{
//    system("rm *.csv");
    if (first)
    {
        mu_ = mu_max;
    }
    else
    {
        mu_ = mu_max2;
    }

    int it = itr_per_mu*(int)(log(mu_/(limMu_) )/log(divFact_));
    int imp = (int)(1/epsilon);
    normalizedDist_.resize(dist_.rows(), 3);
    for(int k = 0; k<dist_.rows(); ++k)
        normalizedDist_.row(k) = dist_.row(k)/dist_.row(k).norm();

    int itr = 0;
    while(imp>(int)(0.25*neighborhood_.rows()) && itr<it)
    {
        if( mu_ > limMu_ && (itr % itr_per_mu) == 0)
            mu_ /= divFact_;

        //compute weighs depending on the current error of the neighbors
        ComputeWeighs();

//        if(itr%10 == 0 || itr<10)
//            save_itr(itr);

        Eigen::Matrix3f C = normalizedDist_.transpose()*weighs_.asDiagonal()*normalizedDist_;
        setNormal(getThirdEigenVector(C));

        imp = 0;
        for (int c = 0; c < neighborhood_.rows(); c++)
        {
            if(sqrt(weighs_(c)) > 0.5)
            {
                ++imp;
            }
        }

        ++itr;

    }

    if (first)
    {
        normalFirst1_ = new Eigen::Vector3f();
        *normalFirst1_ = normal;
    }
    else
    {
        normalSecond1_ = new Eigen::Vector3f();
        *normalSecond1_ = normal;
    }
}

void CApp::OptimizePos(int it)
{

    //as we want to change n and p0, the cost function is not linear with the vector (n,p0)T so we proceed to a gradient descent
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

        for (int c = 0; c < neighborhood_.rows(); c++)
        {
            J(0) = (dist_(c,0) * cos(theta) * cos(phi)    + dist_(c,1) * cos(theta) * sin(phi) + dist_(c,2) * (-sin(theta)));
            J(1) = (dist_(c,0) * sin(theta) * (-sin(phi)) + dist_(c,1) * sin(theta) * cos(phi));
            J(2) = -1;
            JTJ += J * J.transpose();
            JTr += J * (normal.dot(dist_.row(c))-X);
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

    finalNormal_ = normal;
    finalPos_ = pt;
}

void CApp::actualizeMu()
{
    //As the weigh is non linear with mu we proceed to a gradient descent
    std::vector<float> er_tot(neighborhood_.rows());
    ComputeTotalError(er_tot);
    float new_mu = 0;
    float J_mu;
    float J_muTJ_mu = 0;
    float  J_muTr = 0;
    float r;
    float error = 0;
    float tmp_error = 1/epsilon;
    std::vector<float> old_weigh(neighborhood_.rows());
    std::vector<float> er_proj2(neighborhood_.rows());

    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        old_weigh[c] = mu_/(mu_+er_tot[c]*er_tot[c]);
        er_proj2[c] = normal.dot(dist_.row(c))*normal.dot(dist_.row(c));
    }

    int n = 0;
    while (tmp_error-error>lim_error || n<2)
    {
        J_muTJ_mu = 0;
        J_muTr = 0;

        for (int c = 0; c < neighborhood_.rows(); c++)
        {
            J_mu = er_proj2[c]/pow(new_mu + er_proj2[c],2);
            r = new_mu/(new_mu+er_proj2[c]) - old_weigh[c];
            J_muTJ_mu += J_mu * J_mu;
            J_muTr += J_mu * r;
        }

        new_mu += -J_muTr/J_muTJ_mu;

        tmp_error = error;
        error = 0;
        for (int c = 0; c < neighborhood_.rows(); c++)
        {
            r = new_mu / (new_mu + er_proj2[c]) - old_weigh[c];
            error += r*r;
        }
        ++n;
    }

    mu_ = std::max(limMuPos_, new_mu);
}
//for points on edges : can optimize alternatively the normal and the point position

void CApp::OptimizePos1(bool first, float thresh_weigh, float impacter_weigh)
{

    if(!first)
    {
        if(!isSecondOption())
        {
            finalNormal_ = *normalFirst2_;
            finalPos_ = *pointFirst_;
            return;
        }
    }

//    system("rm *.csv");

    //we first actualize mu to be in the same situation as in the output of optimize (because we have changed error evaluation)
    actualizeMu();

    //-----------------------------------------------------------------------------------------

    int it = std::max(itr_per_mu*(int)(log(mu_/(limMuPos_) )/log(divFact_)), itr_min);

    float sum_poids = 0;
    float moy_proj = 0;
    float r;

    for( int itr = 1; itr <= it; ++itr)
    {
//        if(itr%10 == 0 || itr<10)
//            save_itr(itr);

        if( mu_ > limMuPos_ && (itr % itr_per_mu) == 0)
            mu_ /= divFact_;

        //actualize position of the point

        sum_poids = 0;
        moy_proj = 0;

        for (int c = 0; c < neighborhood_.rows(); c++)
        {
            if(sqrt(weighs_(c)) > thresh_weigh)
            {
                r = sqrt(weighs_(c))*normal.dot(dist_.row(c));
                sum_poids += sqrt(weighs_(c));
                moy_proj += r;
            }
        }
        moy_proj = moy_proj/sum_poids;
//        if(noise_ > noise_min)  // to move point only when there is noise
            pt = pt + moy_proj*normal;

        ComputeDist();

        //Compute weighs

        ComputeWeighs_proj();

        //actualize normal

        Eigen::Matrix3f C = dist_.transpose()*weighs_.asDiagonal()*dist_;
        setNormal(getThirdEigenVector(C));
    }

     orient();

     if(first)
     {
         normalFirst2_ = new Eigen::Vector3f();
         *normalFirst2_ = normal;
         pointFirst_ = new Eigen::Vector3f();
         *pointFirst_ = pt;
         evaluate(&impactFirst_, &moyErrorFirst_, impacter_weigh);
     }
     else
     {
         normalSecond2_ = new Eigen::Vector3f();
         *normalSecond2_ = normal;
         pointSecond_ = new Eigen::Vector3f();
         *pointSecond_ = pt;
         evaluate(&impactSecond_, &moyErrorSecond_, impacter_weigh);
         select_normal();
     }
}

//orient normal to the exterior of the edge

void CApp::orient()
{
    float moy_err = 0;
    for (int c = 0; c < neighborhood_.rows(); c++)
    {
        moy_err += normal.dot(dist_.row(c));
    }

    if(moy_err> epsilon)
        normal = -normal;
}

//get actual normal

Eigen::Vector3f CApp::getNormal()
{
    return normal;
}

bool CApp::isSecondOption()
{
    return abs(normalFirst1_->dot(*normalSecond1_))<likelihood_threshold;
}

//optional : set the new point (when optimizing position) as reference to be used in further normal computation

void CApp::setRef( int ref)
{
    ref_ = ref;
    pt = pointcloud_->row(ref);
    ptRef_ = pt;
}

void CApp::reinitPoint()
{
    if(ptRef_(0) != 0)
        setPoint(ptRef_);
}

void CApp::actuNormal( float phi_new, float theta_new)
{
    phi = phi_new;
    theta = theta_new;
    if(abs(theta) < epsilon)
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

void CApp::evaluate(int *impact, float *sum, float impacter_weigh)
{
    float imp = 0;

    float sum_error = 0;
    float sum_poids = 0;

    setPoint(ptRef_);

    for(int c = 0; c<neighborhood_.rows(); ++c)
    {
        if(sqrt(weighs_(c))>impacter_weigh)
        {
//            sum_error += weighs_[c] * normal.dot(dist_[c]);
//            sum_poids += weighs_[c];
            sum_error += normal.dot(dist_.row(c));
            ++sum_poids;
            ++imp;
        }
    }

    *impact = imp;

    if(sum_poids>0)
    {
        *sum = sum_error/sum_poids;
    }
}

//set Normal
void CApp::setNormal(Eigen::Vector3f norm)
{
    normal = norm;
    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));
    if(abs(theta) < epsilon)
        theta = theta_min;
}


//set point as reference

void CApp::setPoint(Eigen::Vector3f point)
{
    pt = point;
    ComputeDist();
}

//Check if result not nan

bool CApp::isNan()
{
    return (finalNormal_(0) != finalNormal_(0) || finalPos_(0) != finalPos_(0));
}

void CApp::select_normal()
{
    if( moyErrorFirst_<moyErrorSecond_ && impactFirst_>(int)(lim_impacters*(float)neighborhood_.rows()) ) // la limite abs(sum_error1) est dégueux A CHANGER
    {
        finalNormal_ = *normalFirst2_;
        finalPos_ = *pointFirst_;
    }
    else if(impactSecond_>(int)(lim_impacters*(float)neighborhood_.rows()))  // la limite abs(sum_error1) est dégueux A CHANGER
    {
        finalNormal_  = *normalSecond2_;
        finalPos_  = *pointSecond_;
    }
    else if(impactFirst_>(int)(lim_impacters*(float)neighborhood_.rows()) )
    {
        finalNormal_  = *normalFirst2_;
        finalPos_  = *pointFirst_;
    }
    else if( moyErrorFirst_<moyErrorSecond_ )
    {
        finalNormal_  = *normalFirst2_;
        finalPos_  = *pointFirst_;
    }
    else
    {
        finalNormal_  = *normalSecond2_;
        finalPos_  = *pointSecond_;
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
        for(int j = 0; j<neighborhood_.rows(); ++j)
            fneigh<<neighborhood_(j,0)<<","<<neighborhood_(j,1)<<","<<neighborhood_(j,2)<<","<<weighs_(j)<<"\n";
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



Eigen::Matrix<float, Eigen::Dynamic, 3> CApp::getNeighborhood()
{
    return neighborhood_;
}
