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


void initFile(std::string file_name);

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

    CApp<Eigen::Vector3f> app;
    int cloud_size = app.Read(argv[1]);
    app.buildTree();

    bool display = true;

    auto t_tot1 = std::chrono::high_resolution_clock::now();
    int n_neigh = atoi(argv[3]);
    float div_fact = atof(argv[4]);
    double mu_init = 1.0;
    double mu_init1 = 0;
    float sum_error = 0;
    float sum_error1 = 0;
    char* output = argv[9];


    std::cout<<"reading reference"<<std::endl<<std::endl;
    std::ifstream fin(argv[10]);
    vector<Eigen::VectorXf> ref;
    if (fin.is_open())
    {
        std::string dechet;
        for (int i=0; i<6; ++i)
        fin>>dechet;

        while (!fin.eof())
        {
            Eigen::VectorXf pt (6);
            for (int i=0; i<pt.size(); ++i)
                fin >> pt(i);
            ref.push_back(pt);
        }
        fin.close();
    }
    else
    {
       std::cout<<"did not find file"<<std::endl<<std::endl;
    }
        float res;
        float res1;
        int n = 0;
        int k = 0;
        int p = 0;
        int o = 0;
        int nan = 0;
        int caca = 0;
        int edge_not_detected = 0;
        int plane_processed = 0;
        float temp = 10000000;
        float noise = atof(argv[5]);
        float resolution = 0.0025;//0.03; // 0.0025;
        float lim_mu = atof(argv[2]);
        float lim_mu_pos = atof(argv[6]);
        float er_supposed;
        float dist_moy = (sqrt((float)(n_neigh))/2.0)*resolution;
        er_supposed = noise/sqrt(noise*noise + dist_moy*dist_moy);

        if(lim_mu < 1e-10)
        {
            lim_mu = noise * (1 + 1/sqrt( (dist_moy/2)*(dist_moy/2) + (2*noise)*(2*noise) ));
        }

        if(lim_mu_pos < 1e-10)
        {
            lim_mu_pos = pow(noise,2);
        }


    initFile(output);
    initFile("wrong_normals.csv");
    initFile("1st_choice.csv");
    initFile("2nd_choice.csv");
    initFile("1st_global_min.csv");
    initFile("2nd_global_max.csv");
    initFile("global_min.csv");
    initFile("true_normals.csv");
    initFile("normals_bords.csv");
    initFile("moy_bord.csv");
    initFile("edge_direction.csv");

    std::cout<<"computing normals"<<std::endl<<std::endl;


    /// --------------------------------------------LOOP------------------------------------------------------------------------------------
    /// ------------------------------------------------------------------------------------------------------------------------------------
    /// ------------------------------------------------------------------------------------------------------------------------------------
    /// ------------------------------------------------------------------------------------------------------------------------------------
    ///

//    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) private(app,n,k,p,o) shared(noise, ref, n_neigh, div_fact, lim_mu, lim_mu_pos it, mu_init)
    for (int i = 0; i < cloud_size; ++i ) // 23467
    {
        Eigen::Vector3f normal= Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_first0 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_first1 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_first2 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_second0 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_second1 = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_second2 = Eigen::Vector3f::Zero();

        Eigen::Vector3f point;
        Eigen::Vector3f point_first = Eigen::Vector3f::Zero();
        Eigen::Vector3f point_second = Eigen::Vector3f::Zero();

        int impact = 0;

        int impact1 = 0;
        app.setRef(i);
        Eigen::Vector3f point_ref = app.getPoint();
        app.selectNeighbors(n_neigh); 
        app.ComputeDist();

        /////////////////////////////////////////////////////////////
        /// PCA for max and min of variance initialization of normal
        ///

         Eigen::Vector3f dir0;
         Eigen::Vector3f dir1;
         Eigen::Vector3f dir2;

         app.pca(dir0, dir1, dir2);

         normal_first0 = dir2;
         app.setNormal(normal_first0);

         /////////////////////////////////////////////////////////

         float moy_error = app.getMoy(); ///////////////////////////////
//         app.writeNormal("global_min.csv");

         n_neigh = app.get_N_neigh();

        if(moy_error > 2*noise + 0.0001) //2*noise -------------------------------------------------------------------------------------------------------
        {
            if(abs(app.getPoint()(0)-1)>0.1 || abs(app.getPoint()(1)-1)>0.1)
                ++plane_processed;
            mu_init = 0;

            res = app.Optimize(div_fact, lim_mu, &mu_init, 1);

            normal_first1 = app.getNormal();

            if(atoi(argv[7]))
            {
//                res = app.Refine(div_fact, lim_mu_pos, &mu_init1);
                res = app.OptimizePos1(div_fact, lim_mu_pos, &mu_init);
            }

            normal_first2 = app.getNormal();
            point_first = app.getPoint();

            float error_thresh = noise+0.001;//(*2) (+0.001)
            app.getImpact(error_thresh, &impact, &sum_error);

            moy_error = sqrt(moy_error/n_neigh);

            //----------------------------------------------------------------------------------------------------

            app.setRef(i);
            app.ComputeDist();
            app.setNormal(normal_first0);

            app.getEdgeDirection(50);

//            app.writeNormal("edge_direction.csv");
            Eigen::Vector3f  normal_temp = app.getNormal().cross(normal_first0);
            normal_second0 = normal_temp/normal_temp.norm();
//            float sens = normal_second0.dot(app.getPoint());
//            if(sens>0)
//                normal_second0=-normal_second0;
            app.setNormal(normal_second0);

            mu_init1 = 1;

            res1 = app.Optimize(div_fact, lim_mu, &mu_init1, 1);

            normal_second1 = app.getNormal();

            if(atoi(argv[7]))
            {
//                    res1 = app.Refine(div_fact, lim_mu_pos, &mu_init1);
                res1 = app.OptimizePos1(div_fact, lim_mu_pos, &mu_init1);
            }

            normal_second2 = app.getNormal();
            point_second = app.getPoint();

            if(normal_second2.dot(normal_first2)>0.9)
                ++p;

            app.getImpact(error_thresh, &impact1, &sum_error1);
            if(point_ref[0]>1.01 && point_ref[0]<1.015 && point_ref[1]<1.007)
                std::cout<<"choisir point : "<<i<<std::endl<<std::endl;
            app.select_normal(&impact, impact1, sum_error, sum_error1, normal_first2, normal_second2, point_first, point_second);
            normal = app.getNormal();
            point = app.getPoint();

            if(impact<0.25*n_neigh)
            {
                ++o;
            }

            if( app.getNormal()(0) == app.getNormal()(0) )
                app.writeNormal(output);
            else
            {
                std::cout<<"nan normal :"<<i<<std::endl;
                ++n;
                ++nan;
            }
        }
        else
        {
            ++k;
            if(abs(app.getPoint()(0)-1)<0.05 && abs(app.getPoint()(1)-1)<0.05)
                ++edge_not_detected;
            if(atoi(argv[7]))
                app.OptimizePos(30);

            normal = app.getNormal();
            point = app.getPoint();

            if( normal(0) == normal(0) )
            {
                app.writeNormal(output);
            }
            else
            {
                std::cout<<"nan normal :"<<i<<std::endl;
                ++n;
                ++nan;
            }

            normal_first1.setZero();
            normal_first2.setZero();
            normal_second0.setZero();
            normal_second1.setZero();
            normal_second2.setZero();
        }

        Eigen::Vector3f normal_ref = {ref[i](3), ref[i](4), ref[i](5)};

//        if( (abs(normal(1))>0.5 && abs(1-point(1))>abs(1-point(0))) || ( abs(normal(0))>0.5 && abs(1-point(0))>abs(1-point(1)) ) || ( abs(normal(0))<0.95 && abs(normal(1)) <0.95 ) )
        if( (abs(normal(1))>0.5 && ref[i](1)>ref[i](0)) || ( abs(normal(0))>0.5 && ref[i](0)>ref[i](1) ) || ( abs(normal(0))<0.98 && abs(normal(1)) <0.98 ) )
        {
            if(display)
                std::cout<<"wrong normal :"<<i<<"          /          "<<n<<std::endl;
            ++n;
            app.writeNormal("wrong_normals.csv");

            app.setNormal(normal_ref);
            app.writeNormal("true_normals.csv");

            app.setNormal(normal_first0);
            app.writeNormal("1st_global_min.csv");

            app.setNormal(normal_second0);
            app.writeNormal("2nd_global_max.csv");

            app.setNormal(normal_first2);
            app.writeNormal("1st_choice.csv");

            app.setNormal(normal_second2);
            app.writeNormal("2nd_choice.csv");
        }

        if( abs(normal(0))<0.98 && abs(normal(1)) <0.98 && display)
        {
            caca++;
            std::cout<<"caca :"<<i<<std::endl;
        }

        if(temp>mu_init)
            temp = mu_init;

        if(i%100==0)
            std::cout<<(float)i/(float)cloud_size*100<<" %"<<std::endl<<std::endl;

        n_neigh = atoi(argv[3]);
    }

      std::cout<<"last µ : "<<temp<<std::endl<<std::endl;

      std::cout<<"number normals for which less than 1/4 neighbors impacted : "<<o<<std::endl<<std::endl;
      std::cout<<"number of points on edge not detected : "<<edge_not_detected<<std::endl<<std::endl;
      std::cout<<"perc of normals which are on plane: "<<(float)k/(float)cloud_size<<std::endl<<std::endl;
      std::cout<<"number of points where global max failed: "<<p<<std::endl<<std::endl;
      std::cout<<"perc normals which have not converged: "<<(double)(k)*100/cloud_size<<" % "<<std::endl<<std::endl;
      std::cout<<"number of caca normals: "<<caca<<std::endl<<std::endl;
      std::cout<<"number of nan normals: "<<nan<<std::endl<<std::endl;
      std::cout<<"number of false normals : "<<n<<std::endl<<"percentage : "<<(float)(n)/(float)cloud_size<<std::endl<<std::endl;

    auto t_tot2 = std::chrono::high_resolution_clock::now();
    std::cout<<"total time to get normals :" <<std::chrono::duration_cast<std::chrono::milliseconds>(t_tot2-t_tot1).count()<<" milliseconds"<<std::endl<<std::endl;


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


