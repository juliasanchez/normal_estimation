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
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#include "app.h"
#include "cloud.h"

const float likelihood_threshold = 0.95; // value for evaluation/comparison between vectors
const float noise_min = 0.0001; // minimum noise to make it not infinite

void initFile(std::string file_name);
void write(std::string file_name, std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f>  normals);
void read(std::string file_name, vector<Eigen::VectorXf>& cloud);

int main(int argc, char *argv[])
{   
    system("rm *.csv");
    if (argc != 11)
    {
        printf("./FastGlobalRegistration [pointcloud] [lim_mu] [neighbors number] [div_fact] [noise] [lim_mu_pos] [optimize_pos?] [with_symmetrics?] [output_file] [Ref]\n");
        return 0;
    }
    else
    {
        for (int i = 1 ; i<argc; i++)
        {
            std::cout<<argv[i]<<"   "<<std::endl<<std::endl;
        }
    }

    std::cout<<"reading reference : "<<argv[1]<<std::endl<<std::endl;
    std::vector<Eigen::VectorXf> ref;
    read(argv[10], ref);

    cloud c;
    int cloud_size = c.Read(argv[1]);
    c.buildTree();
    float resolution = c.getResolution();

    std::vector<Eigen::Vector3f>* pc = c.getPC();
    flann::Index<flann::L2<float>>* tree = c.getTree();
    std::cout<<"pointcloud resolution: "<<resolution<<std::endl<<std::endl;

    int n_neigh = atoi(argv[3]);
    float div_fact = atof(argv[4]);
    char* output = argv[9];

    int n = 0;
    int nan = 0;
    int caca = 0;
    float noise = std::max((float)atof(argv[5]), noise_min);
    float lim_mu = atof(argv[2]);
    float lim_mu_pos = atof(argv[6]);
    float dist_moy = (sqrt((float)(n_neigh))/2.0)*resolution;

    if(lim_mu < epsilon)
    {
//        lim_mu = 0.1; //  angle between pip0 and n à 80° ou 100 ° (10° de ce qui est voulu)
//        lim_mu = noise/(5*resolution);
        lim_mu = noise * (1 + 1/sqrt( (dist_moy/2)*(dist_moy/2) + (2*noise)*(2*noise) ));
//        lim_mu = 0.5*noise * (1 + 1/sqrt( (dist_moy/2)*(dist_moy/2) + (2*noise)*(2*noise) ));
    }

    if(lim_mu_pos < epsilon)
    {
        lim_mu_pos = 0.01*noise;
//        lim_mu_pos = noise*noise;
    }

    std::cout<<"first lim_mu :" <<lim_mu<<std::endl;
    std::cout<<"second lim_mu (lim_mu_pos) :" <<lim_mu_pos<<std::endl<<std::endl;

    std::vector<Eigen::Vector3f> normals(cloud_size);
    std::vector<Eigen::Vector3f> points(cloud_size);
    std::vector<Eigen::Vector3f> wrong_normals;
    std::vector<Eigen::Vector3f> wrong_points;
    std::vector<Eigen::Vector3f> true_normals;
    std::vector<Eigen::Vector3f> true_points;
    std::vector<Eigen::Vector3f> caca_normals;
    std::vector<Eigen::Vector3f> caca_points;

    std::cout<<"computing normals"<<std::endl<<std::endl;
    auto t_tot1 = std::chrono::high_resolution_clock::now();


//    /// --------------------------------------------LOOP------------------------------------------------------------------------------------
//    /// ------------------------------------------------------------------------------------------------------------------------------------
//    /// ------------------------------------------------------------------------------------------------------------------------------------
//    /// ------------------------------------------------------------------------------------------------------------------------------------
//    ///


//    int idx = 7761;
//    for (int i = idx; i < idx+1; ++i )
    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) shared(pc, tree, noise, n_neigh, div_fact, lim_mu, lim_mu_pos, normals, points)
    for (int i = 0; i < cloud_size; ++i)
    {
        Eigen::Vector3f normal_first0 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_first1 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_first2 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_second0 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_second1 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_second2 = Eigen::Vector3f::Zero();

        Eigen::Vector3f point_first = Eigen::Vector3f::Zero();
        Eigen::Vector3f point_second = Eigen::Vector3f::Zero();


        int impact = 0;
        int impact1 = 0;
        float sum_error = 0;
        float sum_error1 = 0;

        CApp app;
        app.setTree(tree);
        app.setPc(pc);
        app.setRef(i);
        Eigen::Vector3f point_ref = app.getPoint();
        app.selectNeighbors(n_neigh);
        app.ComputeDist();
        app.ComputeDistWeighs();

        /////////////////////////////////////////////////////////////

         app.pca();
         normal_first0 = app.getNormal();

         /////////////////////////////////////////////////////

         app.OptimizePos(itr_opti_pos_plan);

         float moy_proj = abs(app.getMoy());

        if( moy_proj > (noise) ) // -------------------------------------------------------------------------------------------------------
        {
            app.setRef(i);
            app.setNormal(normal_first0);
            app.ComputeDist();
            double mu_init = 0;
            app.Optimize(div_fact, lim_mu, &mu_init);

//            app.setNormal(normal_first0);
            bool convert_mu = true;

            normal_first1 = app.getNormal();

            app.OptimizePos1(div_fact, lim_mu_pos, &mu_init, convert_mu);

            normal_first2 = app.getNormal();
            point_first = app.getPoint();

            app.getImpact(&impact, &sum_error);

            app.setRef(i);
            app.ComputeDist();
            app.setNormal(normal_first2);

            app.getEdgeDirection(init2_accuracy);

            Eigen::Vector3f  normal_temp = app.getNormal().cross(normal_first2);
            normal_second0 = normal_temp/normal_temp.norm();
            app.setNormal(normal_second0);

            double mu_init1 = 1;
            app.Optimize(div_fact, lim_mu, &mu_init1);
            convert_mu = true;

//            double mu_init1 = 10*lim_mu_pos;
//            convert_mu = false;

            normal_second1 = app.getNormal();

            if(abs(normal_first1.dot(normal_second1))<likelihood_threshold)
            {
                app.OptimizePos1(div_fact, lim_mu_pos, &mu_init1, convert_mu);
                normal_second2 = app.getNormal();
                point_second = app.getPoint();
                app.getImpact(&impact1, &sum_error1);
                app.select_normal(impact, impact1, sum_error, sum_error1, normal_first2, normal_second2, point_first, point_second);
            }
            else
            {
                app.setNormal(normal_first2);
                app.setPoint(point_first); 
            }

            if( app.getNormal()(0) == app.getNormal()(0) )
                normals[i]=app.getNormal();
            else
            {
                std::cout<<"nan normal :"<<i<<std::endl;
                ++nan;
            }


            if( app.getPoint()(0) == app.getPoint()(0) )
            {
                points[i] = app.getPoint();
            }
            else
            {
                std::cout<<"nan point :"<<i<<std::endl;
                ++nan;
            }

        }
        else
        {
//            app.OptimizePos(itr_opti_pos_plan);

            if( app.getNormal()(0) == app.getNormal()(0) )
            {
                normals[i] = app.getNormal();
            }
            else
            {
                std::cout<<"nan normal :"<<i<<std::endl;
                ++nan;
            }

            if( app.getPoint()(0) == app.getPoint()(0) )
            {
                points[i] = app.getPoint();
            }
            else
            {
                std::cout<<"nan point :"<<i<<std::endl;
                ++nan;
            }
        }

//        points[i] = point_ref;

        if(normals[i].dot(points[i])>0)
            normals[i]=-normals[i];

    }

    auto t_tot2 = std::chrono::high_resolution_clock::now();

//    for (int i = idx; i<idx+1 ; ++i)
    for (int i = 0; i<normals.size(); ++i)
    {
        Eigen::Vector3f point_ref = {ref[i](0), ref[i](1), ref[i](2)};
        Eigen::Vector3f normal_ref = {ref[i](3), ref[i](4), ref[i](5)};
        if(abs(normal_ref.dot(normals[i]))<likelihood_threshold)
        {
            ++n;
            std::cout<<"wrong normal :"<<i<<"          /          "<<n<<std::endl;
            wrong_normals. push_back(normals[i]);
            wrong_points. push_back(points[i]);
            true_normals.push_back(normal_ref);
            true_points.push_back(point_ref);
        }

        if( abs(normal_ref.dot(normals[i]))<likelihood_threshold && abs(normal_ref.dot(normals[i]))>0.2)
        {
            caca++;
            std::cout<<"caca :"<<i<<std::endl;
            caca_normals. push_back(normals[i]);
            caca_points. push_back(points[i]);
        }
    }

    std::cout<<"number of caca normals: "<<caca<<std::endl<<std::endl;
    std::cout<<"number of false normals : "<<n<<"     -------------------    "<<"percentage : "<<(float)(n)/(float)cloud_size<<std::endl<<std::endl;
    std::cout<<"total time to get normals :" <<std::chrono::duration_cast<std::chrono::milliseconds>(t_tot2-t_tot1).count()<<" milliseconds"<<std::endl<<std::endl;

    write(output, points, normals);
    write("wrong.csv", wrong_points, wrong_normals);
    write("caca.csv", caca_points, caca_normals);
    write("true.csv", true_points, true_normals);

    return 0;
}


void initFile(std::string file_name)
{
    std::ofstream fout;
    fout.open(file_name, std::ofstream::trunc);
    if (fout.is_open())
    {
        fout<<"x,y,z,nx,ny,nz\n";
    }
    fout.close();
}


void write(std::string file_name, std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f>  normals)
{
    initFile(file_name);
    std::ofstream fout(file_name, std::ofstream::app);

    if (fout.is_open())
    {
        for (int i = 0; i<normals.size() ; ++i)
        {
            for (int j = 0; j<3; ++j)
            {
//                fout<<ref[i](j)<<",";
                fout<<points[i](j)<<",";
            }
            for (int j = 0; j<3; ++j)
            {
                fout<<normals[i](j)<<",";
            }
            fout<<"\n";
        }

        fout.close();
    }
}

void read(std::string file_name, vector<Eigen::VectorXf>& cloud)
{
    std::ifstream fin(file_name);
    if (fin.is_open())
    {
        string test;
        std::getline ( fin, test, '\n' );

        while (std::getline ( fin, test, ',' ))
        {
            Eigen::VectorXf pt(6);
            pt(0) = stof(test);
            std::getline ( fin, test, ',' );
            pt(1) = stof(test);
            std::getline ( fin, test, ',' );
            pt(2) = stof(test);
            std::getline ( fin, test, ',' );
            pt(3) = stof(test);
            std::getline ( fin, test, ',' );
            pt(4) = stof(test);
            std::getline ( fin, test, '\n' );
            pt(5) = stof(test);

            cloud.push_back(pt);
        }
        fin.close();
    }
    else
    {
       std::cout<<"did not find file"<<std::endl<<std::endl;
    }
}
