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

int main(int argc, char *argv[])
{   
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

    cloud c;
    int cloud_size = c.Read(argv[1]);
    c.buildTree();

    std::vector<Eigen::Vector3f>* pc = c.getPC();
    flann::Index<flann::L2<float>>* tree = c.getTree();

    int n_neigh = atoi(argv[3]);
    float div_fact = atof(argv[4]);
    char* output = argv[9];


    std::cout<<"reading reference"<<std::endl<<std::endl;
    std::ifstream fin(argv[10]);
    vector<Eigen::VectorXf> ref;
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

            ref.push_back(pt);
        }
        fin.close();
    }
    else
    {
       std::cout<<"did not find file"<<std::endl<<std::endl;
    }

        int n = 0;
        int nan = 0;
        int caca = 0;
        float noise = atof(argv[5]);
        float resolution = 0.0025;
        float lim_mu = atof(argv[2]);
        float lim_mu_pos = atof(argv[6]);
        float er_supposed;
        float dist_moy = (sqrt((float)(n_neigh))/2.0)*resolution;
        er_supposed = noise/sqrt(noise*noise + dist_moy*dist_moy);

        if(lim_mu < epsilon)
        {
            lim_mu = noise * (1 + 1/sqrt( (dist_moy/2)*(dist_moy/2) + (2*noise)*(2*noise) ));
        }

        if(lim_mu_pos < epsilon)
        {
            lim_mu_pos = pow(noise,2);
        }

    std::vector<Eigen::Vector3f> normals(cloud_size);
    std::vector<Eigen::Vector3f> points(cloud_size);
    std::vector<Eigen::Vector3f> wrong_normals;
    std::vector<Eigen::Vector3f> wrong_points;

    std::cout<<"computing normals"<<std::endl<<std::endl;
    auto t_tot1 = std::chrono::high_resolution_clock::now();


//    /// --------------------------------------------LOOP------------------------------------------------------------------------------------
//    /// ------------------------------------------------------------------------------------------------------------------------------------
//    /// ------------------------------------------------------------------------------------------------------------------------------------
//    /// ------------------------------------------------------------------------------------------------------------------------------------
//    ///

    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) shared(pc, tree, noise, n_neigh, div_fact, lim_mu, lim_mu_pos, normals, points)
    for (int i = 0; i < cloud_size; ++i )
    {
        CApp app;
        app.setTree(tree);
        app.setPc(pc);
        app.setRef(i);
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
        app.setRef(i);

        app.selectNeighbors(n_neigh);
        app.ComputeDist();

        /////////////////////////////////////////////////////////////

         Eigen::Vector3f dir0;
         Eigen::Vector3f dir1;
         Eigen::Vector3f dir2;

         app.pca(dir0, dir1, dir2);

         normal_first0 = dir2;
         app.setNormal(normal_first0);

         /////////////////////////////////////////////////////

        if(app.getMoy() > 1.5*noise + noise_min) // -------------------------------------------------------------------------------------------------------
        {
            double mu_init = 0;
            app.Optimize(div_fact, lim_mu, &mu_init);

            normal_first1 = app.getNormal();

            app.OptimizePos1(div_fact, lim_mu_pos, &mu_init);

            normal_first2 = app.getNormal();
            point_first = app.getPoint();

            app.getImpact(&impact, &sum_error);

            //----------------------------------------------------------------------------------------------------

            app.setRef(i);
            app.ComputeDist();
            app.setNormal(normal_first2);

            app.getEdgeDirection(init2_accuracy);

            Eigen::Vector3f  normal_temp = app.getNormal().cross(normal_first2);
            normal_second0 = normal_temp/normal_temp.norm();
            app.setNormal(normal_second0);

            double mu_init1 = 1;

            app.Optimize(div_fact, lim_mu, &mu_init1);

            normal_second1 = app.getNormal();

            if(abs(normal_first1.dot(normal_second1))<likelihood_threshold)
            {
                app.OptimizePos1(div_fact, lim_mu_pos, &mu_init1);
                normal_second2 = app.getNormal();
                point_second = app.getPoint();
                app.getImpact(&impact1, &sum_error1);
                app.select_normal(&impact, impact1, sum_error, sum_error1, normal_first2, normal_second2, point_first, point_second);
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
            app.OptimizePos(itr_opti_pos_plan);

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
    }

    auto t_tot2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i<normals.size() ; ++i)
    {
        Eigen::Vector3f normal_ref = {ref[i](3), ref[i](4), ref[i](5)};
        if(abs(normal_ref.dot(normals[i]))<likelihood_threshold)
        {
            ++n;
            std::cout<<"wrong normal :"<<i<<"          /          "<<n<<std::endl;
            wrong_normals. push_back(normals[i]);
            wrong_points. push_back(points[i]);
        }

        if( abs(normals[i](0))<0.99 && abs(normals[i](1)) <0.99)
        {
            caca++;
            std::cout<<"caca :"<<i<<std::endl;
        }
    }

    std::cout<<"number of caca normals: "<<caca<<std::endl<<std::endl;
    std::cout<<"number of false normals : "<<n<<"     -------------------    "<<"percentage : "<<(float)(n)/(float)cloud_size<<std::endl<<std::endl;
    std::cout<<"total time to get normals :" <<std::chrono::duration_cast<std::chrono::milliseconds>(t_tot2-t_tot1).count()<<" milliseconds"<<std::endl<<std::endl;

    write(output, points, normals);
    write("wrong.csv", wrong_points, wrong_normals);

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
