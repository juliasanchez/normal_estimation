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
// furnished to do so, subject to the following conditions
// ----------------------------------------------------------------------------

#include "app.h"


// reads file and puts it into pointcloud_

template<typename vec>
int CApp<vec>::Read(const char* filepath)
{
        std::cout<<"reading file"<<std::endl<<std::endl;
    std::ifstream fin(filepath);
    vector<vec> res;
    int n = 0;
    if (fin.is_open())
    {
        while (!fin.eof())
        {
            vec pt;
            for (int i=0; i<pt.size(); ++i)
                fin >> pt(i);
            res.push_back(pt);
            ++n;
        }
        fin.close();
        pointcloud_ = res;
        pt = pointcloud_[5];
    }
    else
    {
       std::cout<<"did not find file"<<std::endl<<std::endl;
    }
    return n;
}



//build tree to compute neighbors

template<typename vec>
void CApp<vec>::buildTree()
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


// Get neighbors from tree and puts it in neighborhood_

template<typename vec>
void CApp<vec>::selectNeighbors(int neigh_number)
{
    std::vector<float> dis;
    std::vector<int> neigh;

    SearchFLANNTree(tree_, pt, neigh, dis, neigh_number);
    neighborhood_.resize (neigh.size()-1);

    for (int i = 1; i < neigh.size(); ++i)
    {
        vec test = pointcloud_[neigh[i]]-pt;
        if(test.norm() != 0)
            neighborhood_[i-1] = pointcloud_[neigh[i]];
        else
        {
            SearchFLANNTree(tree_, pt, neigh, dis, neigh_number+1);
            neigh.erase(neigh.begin()+i -1);
            i = i-1;
        }
    }
}


//get neighborhood size

template<typename vec>
int CApp<vec>::get_N_neigh()
{
    return neighborhood_.size();
}

//Search function in the tree to get the nearest. (internally used in selectNeighbors)

template<typename vec>
void CApp<vec>::SearchFLANNTree(flann::Index<flann::L2<float>>* index,
                            vec& input,
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
template<typename vec>
void CApp<vec>::ComputeDist()
{
    dist_.resize (neighborhood_.size());
    for (int i = 0; i < neighborhood_.size(); ++i)
    {
        dist_[i] = neighborhood_[i] - pt;
        moy_ += dist_[i];
    }

    moy_ = moy_/neighborhood_.size();
}

template<typename vec>
void CApp<vec>::initNormal()
{
    normal(0) = sin(theta) * cos(phi);
    normal(1) = sin(theta) * sin(phi);
    normal(2) = cos(theta);
}



template<typename vec>
void CApp<vec>::pca(Eigen::Vector3f &dir0, Eigen::Vector3f &dir1, Eigen::Vector3f &dir2)
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

template<typename vec>
void CApp<vec>::getEdgeDirection(int it)
{
    phi= atan2(normal(1),normal(0));
    theta= acos(normal(2));

    initNormal();
    Eigen::Vector3f normal_test, ran, global_min;

    global_min = normal.segment(0,3);

    ran(0) = 0.5;
    ran(1) = 0;
    ran(2) = 0.86602540378;

    int N_hist = 10;
    Eigen::MatrixXf hist(N_hist,N_hist);

    normal_test = ran.cross(global_min);
    normal_test /= normal_test.norm();

    theta = 2*M_PI/it;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate (Eigen::AngleAxisf (theta, global_min));

    float temp = 1000000;

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


//template<typename vec>
//void CApp<vec>::ComputeWeighs(bool normalize, double mu, bool use_last)
//{
//    for (int c = 0; c < neighborhood_.size(); c++) {

//        vec normalized;
//        if(normalize)
//            normalized = dist_[c]/dist_[c].norm();
//        else
//            normalized = dist_[c];
//        float er = normal.dot(normalized) * normal.dot(normalized);


//        if(use_last)
//        {
//            float diff = poids_[c] - ( mu / (er + mu) );

//            if(diff_poids_[c]*diff>0)
//            {
//                poids_[c] = mu / (er + mu);
//                diff_poids_[c] = diff;
//            }

//        }
//        else
//        {
//            if(poids_[c] != 0)
//                diff_poids_[c] = poids_[c] - ( mu / (er + mu) );

//            poids_[c] = mu / (er + mu);
//        }
//    }

//    float max_poids = *std::max_element(poids_.begin(), poids_.end());

//    for (int c = 0; c < neighborhood_.size(); c++) {
//        poids_[c] /= max_poids;
//    }

//}

template<typename vec>
void CApp<vec>::ComputeWeighs(bool normalize, double mu, bool use_last)
{
    std::vector<float> er_angle(neighborhood_.size());
    std::vector<float> er_proj(neighborhood_.size());

    for (int c = 0; c < neighborhood_.size(); c++) {

        vec normalized;
        normalized = dist_[c]/dist_[c].norm();
//        er_angle[c] = normal.dot(normalized) * normal.dot(normalized);
//        er_proj[c] = normal.dot(dist_[c]) * normal.dot(dist_[c]);
        er_angle[c] = abs(normal.dot(normalized));
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


    for (int c = 0; c < neighborhood_.size(); c++)
    {
        float alpha = ( max_dist-dist_[c].norm() )/max_dist;
        float er_tot = (1-alpha)*er_angle[c]/er_angle_max + (1-alpha) * er_proj[c]/er_proj_max ;
        er_tot = er_tot*er_tot;

        if(use_last)
        {
            float diff = poids_[c] - ( mu / (er_tot + mu) );

            if(diff_poids_[c]*diff>0)
            {
                poids_[c] = mu / (er_tot + mu);
                diff_poids_[c] = diff;
            }

        }
        else
        {
            if(poids_[c] != 0)
                diff_poids_[c] = poids_[c] - ( mu / (er_tot + mu) );

            poids_[c] = mu / (er_tot + mu);
        }
    }

    float max_poids = *std::max_element(poids_.begin(), poids_.end());

    for (int c = 0; c < neighborhood_.size(); c++) {
        poids_[c] /= max_poids;
    }

}


//Once the global minimum of the function is computed we make the minimization WEIGHTED depending on the neighbors error

template<typename vec>
float CApp<vec>::Optimize(float div_fact, float lim_mu, double* mu_init, bool normalize) // normalize decided if minimiser projection ou minimiser angle : projection normalisée
//probleme with angle : points on the same z have different errors : less error for those far from p0 in z direction
//problem with projection : points on the same p0pi direction have different errors : less error for closest points.
{

    bool display = false;
    if (display)
    {
        system("rm neighbors*.csv");
        system("rm normal*.csv");
    }
    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));

    double mu = *mu_init;

    if (mu < 1e-10 && normalize)
    {
//        mu = 5.0;//----------------------------------------------------------------------------------------------
        mu = 1;
    }
    else if (  ( (mu-1.0) < 1e-10 ) && normalize)
    {
        std::vector<float> dist(dist_.size());
        for (int i = 0; i<dist.size(); ++i)
        {
            vec normalized = dist_[i]/dist_[i].norm();
            dist[i] = normal.dot(normalized) * normal.dot(normalized);
        }

        std::sort(dist.begin(), dist.end());
        mu = dist[dist.size()-1];
//        mu = 0.1*dist[dist.size()-1];
//        mu = dist[0]; //----------------------------------------------------------------------------------------------
    }
    else if (mu < 1e-10 && !normalize)
    {
        std::vector<float> dist(dist_.size());
        for (int i = 0; i<dist.size(); ++i)
        {
            vec normalized = dist_[i];
            dist[i] = normal.dot(normalized) * normal.dot(normalized);
        }

        std::sort(dist.begin(), dist.end());
        mu = 2*dist[dist.size()-1];
    }
    else if ( ( (mu-1.0) < 1e-10 ) && !normalize)
    {
        std::vector<float> dist(dist_.size());
        for (int i = 0; i<dist.size(); ++i)
        {
            vec normalized = dist_[i];
            dist[i] = normal.dot(normalized) * normal.dot(normalized);
        }

        std::sort(dist.begin(), dist.end());
        mu = dist[dist.size()-1];
    }


    int itr = 1;
    float error = 1000000;
    float diff = 1000000;
    float tmp = 0;
    poids_.resize(neighborhood_.size());
    diff_poids_.resize(neighborhood_.size());
    error_.resize(neighborhood_.size());
    diff_error_.resize(neighborhood_.size());
    for (int c = 0; c < neighborhood_.size(); c++) {
        poids_[c] = 0;
        diff_poids_[c] = 0;
        error_[c] = 0;
        diff_error_[c] = 0;
    }

    int N = 2;
    int it = N*(int)(std::log(mu/lim_mu)/std::log(div_fact));

    while ( (itr <= it + 20 ) || itr<20) //&& diff>0.0000001  CHANGEMENT DE LA LIMITE POUR ALLER PLUS LOIN APRES AVOIR ATTEINT LIM_MU
    {
        if( mu > lim_mu && (itr % N) == 0)
            mu /= div_fact;

        const int nvariable = pt.size()-1;	// dimensions of J
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        double r;

        error = 0;
        float sum_poids = 0;

        float moy_posx = 0;
        float moy_posy = 0;
        float moy_proj = 0;

        //actualize normal

        //////////////////////////////////////////////////////////////////////////////////////////////////////
          ComputeWeighs(normalize, mu, false);
        //////////////////////////////////////////////////////////////////////////////////////////////////////

        //DISPLAY------------------------------------------------------------------------------------------------------------
        if( ( itr % 10 == 0 || itr < 10 ) && display)
        {
            std::ofstream fneigh;
            std::stringstream stm;
            stm.str("");
            stm<<"neighbors"<<itr<<".csv";
            std::string neighbors_file = stm.str();
            fneigh.open(neighbors_file, std::ofstream::trunc);
            if (fneigh.is_open())
            {
                fneigh<<"x,y,z,weight\n";
            }
            fneigh.close();
            writeNeighbors(neighbors_file);

            std::ofstream fnormal;
            stm.str("");
            stm<<"normal"<<itr<<".csv";
            std::string normal_file = stm.str();
            fnormal.open(normal_file, std::ofstream::trunc);
            if (fnormal.is_open())
            {
                fnormal<<"x,y,z,nx,ny,nz,iteration\n";
                fnormal<<pt(0)<<","<<pt(1)<<","<<pt(2)<<","<<normal(0)<<","<<normal(1)<<","<<normal(2)<<","<<(float)(itr)/(float)(it)<<"\n";
            }
            fnormal.close();
            writeNeighbors(neighbors_file);

        }

        //---------------------------------------------------------------------------------------------------------------------

        for (int c = 0; c < neighborhood_.size(); c++) {

            vec normalized = dist_[c]/dist_[c].norm();
            J(0) = poids_[c] * (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
            J(1) = poids_[c] * (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
            r = poids_[c] * normal.dot(normalized);
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        theta = theta + result(0);
        phi = phi + result(1);

        normal(0) = sin(theta) * cos(phi);
        normal(1) = sin(theta) * sin(phi);
        normal(2) = cos(theta);

        for (int c = 0; c < neighborhood_.size(); c++) {
            ////////////////////////////////////////
            //valeurs affichées pour débugging
            error = error +r*r;

            moy_posx += poids_[c]*neighborhood_[c](0);
            moy_posy += poids_[c]*neighborhood_[c](1);
            moy_proj +=r;
            ////////////////////////////////////////

            sum_poids += poids_[c];
        }

        moy_posx /= sum_poids;
        moy_posy /= sum_poids;
        moy_proj /= sum_poids;

        error = sqrt(error / sum_poids);

        if(itr != 1)
            diff = abs(tmp - error);

        ++itr;

        tmp = error;
    }

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));
    //theta = atan2(normal(2), sqrt(normal(0)*normal(0)+normal(1)*normal(1)));

    *mu_init = mu;
    return error;
}


//optional : for points which are in the middle of planes (do not need a weighted minimization) : can optimize alternatively the normal and the point position

template<typename vec>
float CApp<vec>::OptimizePos(int it)
{
    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));

    float X = 0;

    int itr = 1;
    float error = 1000000;
    float diff = 1000000;
    float tmp = 0;


    while ( (itr <= it && error>0.00001 && diff>0.0000001) || itr<20)
    {
        const int nvariable = 3;
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        double r;
        error = 0;
        float moy_proj = 0;
        float sum_poids = 0;

        //actualize normal

        for (int c = 0; c < neighborhood_.size(); c++) {

            vec normalized = dist_[c];
            J(0) = (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
            J(1) = (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
            J(2) = -1; //( temp/dist_[c].norm() ) * ( pow(normal.dot(normalized),2) - 1 );
            r = normal.dot(normalized)-X;
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        theta = theta + result(0);
        phi = phi + result(1);
        X += result(2);

        normal(0) = sin(theta) * cos(phi);
        normal(1) = sin(theta) * sin(phi);
        normal(2) = cos(theta);

        for (int c = 0; c < neighborhood_.size(); c++) {

            vec normalized = dist_[c]-X*normal;
            r = normal.dot(normalized);
            error = error + r*r;
            moy_proj += r;
        }

        moy_proj /= neighborhood_.size();

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

    return error;
}


//for points on edges : can optimize alternatively the normal and the point position (some points are falsely optimized because of the precision of the normal)

template<typename vec>
float CApp<vec>::OptimizePos1(float div_fact, float lim_mu, double* mu_init)
{
    bool display = false;
    if (display)
    {
        system("rm neighbors*.csv");
        system("rm normal*.csv");
    }

    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));

    //si j'utilise le calcul normalisé dans optimize je dois adapter mu :
    float temp_i;
    float tempo = 100000000;

    double mu = *mu_init;

    for (int i = 0; i<dist_.size(); ++i)
    {
        if(tempo>abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2)))
        {
            tempo = abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2));
            temp_i = i;
        }
    }

//    mu = 5*std::max(lim_mu,(float)(pow(normal.dot(dist_[temp_i]),2))); //------------------------------------------------------------------------
    mu = 0.5*std::max(lim_mu,(float)(pow(normal.dot(dist_[temp_i]),2)));
    *mu_init = mu;

    std::vector<float> weigh(neighborhood_.size());
    float sigma = 0.1;
    float sum=0;

    for (int c = 0; c < neighborhood_.size(); c++) {

        weigh[c] = exp(-dist_[c].dot(dist_[c])/(2*sigma*sigma));
        sum = sum + weigh[c];
    }

    int itr = 1;
    float error = 1000000;
    float diff = 1000000;
    float tmp = 0;
    int p=0;
    int N = 3;//------------------------------------------------------------------------

    int it = N*(int)(log(mu/(lim_mu) )/log(div_fact));

    while ( (itr <= it && p<5) || itr<21)//&& error>0.000001
    {

        //DISPLAY------------------------------------------------------------------------------------------------------------
        if( ( itr % 10 == 0 || itr < 10 ) && display)
        {
            std::ofstream fneigh;
            std::stringstream stm;
            stm.str("");
            stm<<"neighbors"<<itr<<".csv";
            std::string neighbors_file = stm.str();
            fneigh.open(neighbors_file, std::ofstream::trunc);
            if (fneigh.is_open())
            {
                fneigh<<"x,y,z,weight\n";
            }
            fneigh.close();
            writeNeighbors(neighbors_file);

            std::ofstream fnormal;
            stm.str("");
            stm<<"normal"<<itr<<".csv";
            std::string normal_file = stm.str();
            fnormal.open(normal_file, std::ofstream::trunc);
            if (fnormal.is_open())
            {
                fnormal<<"x,y,z,nx,ny,nz,iteration\n";
                fnormal<<pt(0)<<","<<pt(1)<<","<<pt(2)<<","<<normal(0)<<","<<normal(1)<<","<<normal(2)<<","<<(float)(itr)/(float)(it)<<"\n";
            }
            fnormal.close();
        }

        //---------------------------------------------------------------------------------------------------------------------

        if( mu > lim_mu && (itr % N) == 0)
            mu /= div_fact;

        //actualize position of the point

        float sum_poids = 0;
        float moy_posx = 0;
        float moy_posy = 0;
        float moy_proj = 0;

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            vec normalized = dist_[c];
            float er = normal.dot(dist_[c])*normal.dot(dist_[c]);
            if(poids_[c] > 0.7)
            {
                double r = poids_[c]*normal.dot(normalized);
                sum_poids += poids_[c];
                moy_posx += poids_[c]*neighborhood_[c](0);
                moy_posy += poids_[c]*neighborhood_[c](1);
                moy_proj +=r;
            }
        }
        moy_posx = moy_posx/sum_poids;
        moy_posy = moy_posy/sum_poids;
        moy_proj = moy_proj/sum_poids;
        pt = pt + moy_proj*normal;

        ComputeDist();

        //Compute weighs

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            float er = normal.dot(dist_[c])*normal.dot(dist_[c]);
            weigh[c] = 1;
            poids_[c] = sqrt(weigh[c])*mu / (er + mu);
        }

        float max_poids = *max_element(poids_.begin(), poids_.end());

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            poids_[c] /= max_poids;
        }

        //actualize normal

        const int nvariable = pt.size()-1;	// dimensions of J
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            vec normalized = dist_[c];
            J(0) = poids_[c] * (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
            J(1) = poids_[c] * (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
            double r = poids_[c] * normal.dot(normalized);
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        theta = theta + result(0);
        phi = phi + result(1);

        normal(0) = sin(theta) * cos(phi);
        normal(1) = sin(theta) * sin(phi);
        normal(2) = cos(theta);

        //Compute error

        error = 0;
        for (int c = 0; c < neighborhood_.size(); c++) {
            double r = poids_[c]*normal.dot(dist_[c]);
            sum_poids += poids_[c];
            error = error + r*r;
        }


        error = sqrt(error / sum_poids);

        if(itr != 1)
            diff = abs(tmp - error);

        ++itr;

        tmp = error;

        if(diff<0.0000001)
                ++p;
        else
            p = 0;

    }

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    *mu_init = mu;

    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));

    return error;
}


//optional : for points which are in the middle of planes (do not need a weighted minimization) : can optimize alternatively the normal and the point position

template<typename vec>
float CApp<vec>::OptimizePos2(float div_fact, float lim_mu, double* mu_init)
{
    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));

    //si j'utilise le calcul normalisé dans optimize je dois adapter mu : ------------------------------------------------------------------------------------------------------

    float temp_i;
    float tempo = 100000000;

    double mu = *mu_init;

    for (int i = 0; i<dist_.size(); ++i)
    {
        if(tempo>abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2)))
        {
            tempo = abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2));
            temp_i = i;
        }
    }

    mu = 0.5*std::max(lim_mu,(float)(pow(normal.dot(dist_[temp_i]),2)));
    *mu_init = mu;

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    float X = 0;

    int itr = 1;
    float error = 1000000;
    float diff = 1000000;
    float tmp = 0;
    int N = 5;
    int p =0;

    int it = N*(int)(log(mu/(lim_mu) )/log(div_fact));

    while ( (itr <= it && p<5) || itr<20)
    {
        if( mu > lim_mu && (itr % N) == 0)
            mu /= div_fact;

        const int nvariable = 3;
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        double r;
        error = 0;

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            float er = normal.dot(dist_[c])*normal.dot(dist_[c]);
            poids_[c] = mu / (er + mu);
        }

        float max_poids = *max_element(poids_.begin(), poids_.end());

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            poids_[c] /= max_poids;
        }

        //actualize normal and point position

        for (int c = 0; c < neighborhood_.size(); c++)
        {
            J(0) = poids_[c]*(dist_[c](0) * cos(theta) * cos(phi)    + dist_[c](1) * cos(theta) * sin(phi) + dist_[c](2) * (-sin(theta)));
            J(1) = poids_[c]*(dist_[c](0) * sin(theta) * (-sin(phi)) + dist_[c](1) * sin(theta) * cos(phi));
            J(2) = -poids_[c];
            r = poids_[c] * normal.dot(dist_[c])-X;
            JTJ += J * J.transpose();
            JTr += J * r;
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        theta = theta + result(0);
        phi = phi + result(1);
        X += result(2);;

        normal(0) = sin(theta) * cos(phi);
        normal(1) = sin(theta) * sin(phi);
        normal(2) = cos(theta);

        //actualize error

        float sum_poids = 0;
        for (int c = 0; c < neighborhood_.size(); c++)
        {
            vec normalized = dist_[c]-X*normal;
            r = poids_[c] * normal.dot(normalized);
            error = error + r*r;
            sum_poids += poids_[c];
        }

        error = sqrt(error / sum_poids);

        if(itr != 1)
            diff = abs(tmp - error);

        if(diff<0.0000001)
            ++p;
        else
            p = 0;

        ++itr;

        tmp = error;

    }

    pt = pt + X*normal;
    ComputeDist();

    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    return error;
}


template<typename vec>
float CApp<vec>::Refine(float div_fact, float lim_mu, double* mu_init)
{
    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));
    //theta = atan2(normal(2), sqrt(normal(0)*normal(0)+normal(1)*normal(1)));

    double mu = *mu_init;

    //si j'utilise le calcul normalisé dans optimize je dois adapter mu :
    float temp_i;
    float tempo = 100000000;

    for (int i = 0; i<dist_.size(); ++i)
    {
        if(tempo>abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2)))
        {
            tempo = abs(mu-pow(normal.dot(dist_[i]/dist_[i].norm()),2));
            temp_i = i;
        }
    }

    mu = std::max(lim_mu,(float)(pow(normal.dot(dist_[temp_i]),2)));
    *mu_init = mu;

    float X = 0;

    std::vector<float> weigh(neighborhood_.size());
    float sigma = 0.1;
    float sum=0;
    poids_.resize(neighborhood_.size());

    for (int c = 0; c < neighborhood_.size(); c++) {

        weigh[c] = exp(-dist_[c].dot(dist_[c])/(2*sigma*sigma));
        sum = sum + weigh[c];
    }

//    for (int c = 0; c < neighborhood_.size(); c++) {
//        float er = normal.dot(dist_[c])*normal.dot(dist_[c]);
//        poids_[c] = sqrt(weigh[c])*mu / (er+ mu);
//    }

    int itr = 1;
    float error = 1000000;
    float diff = 1000000;
    float tmp = 0;
    int p=0;

    int it = 2*(int)(log(mu/(lim_mu) )/log(div_fact));

    while ( (itr <= it && p<3) || itr<20)//&& error>0.000001
    {
        X=0;
        if( mu > lim_mu && (itr % 2) == 0)
            mu /= div_fact;

        const int nvariable = pt.size()-1;	// dimensions of J
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        double r;
        vec dist_moy = vec::Zero();
        float sum_poids = 0;
        error = 0;

        //compute weigthed average point of the neighborhood

        for (int c = 0; c < neighborhood_.size(); c++) {
                dist_moy += poids_[c]*dist_[c];
                sum_poids += poids_[c];
        }

        dist_moy /= sum_poids;

        for (int c = 0; c < neighborhood_.size(); c++) {
                vec normalized = dist_[c]-dist_moy;
                J(0) = poids_[c] * (normalized(0) * cos(theta) * cos(phi)    + normalized(1) * cos(theta) * sin(phi) + normalized(2) * (-sin(theta)));
                J(1) = poids_[c] * (normalized(0) * sin(theta) * (-sin(phi)) + normalized(1) * sin(theta) * cos(phi));
                r = poids_[c] * normal.dot(normalized);
                JTJ += J * J.transpose();
                JTr += J * r;
                error += r*r ;

                float er = normal.dot(normalized)*normal.dot(normalized);
                poids_[c] = sqrt(weigh[c])*mu / (er+ mu);
        }

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        theta = theta + result(0);
        phi = phi + result(1);

        normal(0) = sin(theta) * cos(phi);
        normal(1) = sin(theta) * sin(phi);
        normal(2) = cos(theta);

        error = sqrt(error / sum_poids);
        if(itr != 1)
            diff = abs(tmp - error);

        tmp = error;

        if(diff<0.0000001)
                ++p;

        ++itr;
    }


    float sens = normal.dot(pt);
    if(sens>0)
        normal=-normal;

    *mu_init = mu;

    phi = atan2(normal(1),normal(0));
    theta = acos(normal(2));

    return error;
}

//get average point of the neighborhood

template<typename vec>
vec CApp<vec>::getVecMoy()
{
    return moy_;
}


//get actual normal

template<typename vec>
vec CApp<vec>::getNormal()
{
    return normal;
}


//optional : set the new point (when optimizing position) as reference to be used in further normal computation (better if not needed

template<typename vec>
void CApp<vec>::setRef( int ref)
{
    ref_ = ref;
    pt = pointcloud_[ref];
}


//get actual point

template<typename vec>
vec CApp<vec>::getPoint()
{
    return pt;
}


// put the actual error (projection of p_ip_0 on normal) of each point in a vector in parameters

template<typename vec>
void CApp<vec>::getError(std::vector<float>* error)
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

template<typename vec>
void CApp<vec>::getErrorNormalized(std::vector<float>* error)
{
    std::vector<float> err(neighborhood_.size());
    ComputeDist();
    for (int c = 0; c < neighborhood_.size(); c++)
    {
        vec normalized = dist_[c]/dist_[c].norm();
        err[c] = normal.dot(normalized);
    }
    *error=err;
}


//compute average projection of p_ip_0 on normal

template<typename vec>
float CApp<vec>::getMoy()
{
    ComputeDist();
    vec pt_moy = vec::Zero();
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

template<typename vec>
void CApp<vec>::getImpact(float error_thresh, int *impact, float *sum)
{
    float imp = 0;
    std::vector<float> error;

    getError(&error);

    float sum_error = 0;
    float sum_poids = 0;

    for(int j = 0; j<neighborhood_.size(); ++j)//---------------------------------------------------------------------------------------------
    {
        if(poids_[j]>0.6)
        {
            sum_error += poids_[j] * error[j];
            sum_poids += poids_[j];
            ++imp;
        }
    }

    *impact = imp;
//    *sum = sum_error/imp;
    if(sum_poids>0)
    {
        *sum = sum_error/sum_poids;
    }
}



//writeNormal in file

template<typename vec>
void CApp<vec>::writeNormal(const char* filepath)
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

template<typename vec>
void CApp<vec>::writeNeighbors(std::string filepath)
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

template<typename vec>
void CApp<vec>::writeErrors(std::string filepath)
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

template<typename vec>
void CApp<vec>::setNormal(vec norm)
{
    normal = norm;
}


//set point as reference

template<typename vec>
void CApp<vec>::setPoint(vec point)
{
    pt = point;
}

template<typename vec>
void CApp<vec>::select_normal(int* impact, int impact1, float sum_error, float sum_error1, vec& normal_first2, vec& normal_second2, vec& point_first, vec& point_second)
{
//    if(impact1-*impact>15)  //res1<res && impact1>impact sum_error>sum_error1 sup1<sup  // si l'erreur des points qui participent est plus faible dans la deuxième séparation
//    {
//        normal = normal_second2;
////        *impact = impact1;
//    }
//    else if (*impact-impact1>15)
//    {
//        normal = normal_first2;
//        setNormal(normal);
//    }
//    else if(sum_error1>sum_error)
//    {
//        normal = normal_first2;
//        setNormal(normal);
//    }
//    else
//    {
//        normal = normal_second2;
////        *impact = impact1;
//    }
////-----------------------------------------------------------------------------------------------------------------------
//    if(*impact<(int)(0.1*neighborhood_.size()))
//    {
//        normal = normal_second2;
//        setPoint(point_second);
//    }
//    else if (impact1<(int)(0.1*neighborhood_.size()))
//    {
//        normal = normal_first2;
//        setNormal(normal);
//        setPoint(point_first);
//    }
//    else if(sum_error1>sum_error)
//    {
//        normal = normal_first2;
//        setNormal(normal);
//        setPoint(point_first);
//    }
//    else
//    {
//        normal = normal_second2;
//        setPoint(point_second);
//    }

    //-----------------------------------------------------------------------------------------------------------------------
        if(sum_error1>sum_error)
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


template<typename vec>
std::vector<vec> CApp<vec>::getNeighborhood()
{
    return neighborhood_;
}

template class CApp<Eigen::Vector2f>;
template class CApp<Eigen::Vector3f>;
