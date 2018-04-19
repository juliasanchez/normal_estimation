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

void initFile(std::string );
void write(std::string , std::vector<Eigen::Vector3f> , std::vector<Eigen::Vector3f> );
void read(std::string , vector<Eigen::VectorXf>& );
std::string create_output_name(std::string, int);

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

    //------------------------------------------------------------------------------------------------

    cloud c;
    std::string input_name = argv[1];
    int cloud_size = c.Read(input_name);
    c.buildTree();
    float resolution = c.getResolution();

    std::vector<Eigen::Vector3f>* pc = c.getPC();
    flann::Index<flann::L2<float>>* tree = c.getTree();
    std::cout<<"pointcloud resolution: "<<resolution<<std::endl<<std::endl;

    int n_neigh = atoi(argv[3]);
    float div_fact = atof(argv[4]);

    std::string output = create_output_name(input_name, n_neigh);

    int n = 0;
    int nan = 0;
    int caca = 0;
    float noise = (float)atof(argv[5]);
    noise = std::max(noise, noise_min);
    noise = std::min(noise, noise_max);

    float lim_mu_pos = 0.01*noise;

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


//    int idx = 28988;
//    for (int i = idx; i < idx+1; ++i )
    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) shared(pc, tree, noise, n_neigh, div_fact, lim_mu_pos, normals, points)
    for (int i = 0; i < cloud_size; ++i)
    {
        CApp app(pc, tree, i, noise);
        Eigen::Vector3f point_ref = app.getPoint();
        app.selectNeighbors(n_neigh);

        app.init1();

        app.OptimizePos(itr_opti_pos_plan);

        if( app.isOnEdge() ) // when the projection error is greater than the noise-------------------------------------------------------------------------------------------------------
        {
            //Compute first solution ------------------------------------------------------------------------------------------------------

            app.reinitPoint();
            app.reinitFirst0();
            app.setParams(div_fact, lim_mu, lim_mu_pos);

            bool first = true;
            app.Optimize(first);
            app.OptimizePos1(first);

            //Compute second solution ------------------------------------------------------------------------------------------------------

            app.reinitPoint();

            app.init2();
            first = false;
            app.Optimize(first);
            app.OptimizePos1(first);

            //save result ------------------------------------------------------------------------------------------------------

            if( !app.isNan())
            {
                normals[i] = app.finalNormal_;
                points[i] = app.finalPos_;
            }
            else
            {
                normals[i]={0,0,0};
                points[i] = {0,0,0};
                std::cout<<"nan normal or point :"<<i<<std::endl;
                ++nan;
            }
        }
        else
        {
            if( !app.isNan())
            {
                normals[i] = app.finalNormal_;
                points[i] = app.finalPos_;
            }
            else
            {
                std::cout<<"nan normal or point :"<<i<<std::endl;
                ++nan;
            }
        }

//        points[i] = point_ref;

        if(normals[i].dot(points[i])>0)
            normals[i]=-normals[i];

    }

    auto t_tot2 = std::chrono::high_resolution_clock::now();

    std::cout<<"reading reference : "<<input_name<<std::endl<<std::endl;
    std::vector<Eigen::VectorXf> ref;
    read(argv[10], ref);

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








//____________________________________________________________- I/O FUNCTIONS -___________________________________________________________










std::string create_output_name(std::string input_name, int n_neigh)
{
    size_t lastindex_point = input_name.find_last_of(".");
    lastindex_point -= 6;
    size_t lastindex_slash = input_name.find_last_of("/");
    if (lastindex_slash==std::string::npos)
    {
       lastindex_slash = 0;
    }

    input_name = input_name.substr(lastindex_slash+1, lastindex_point-(lastindex_slash+1));
    std::stringstream stm;
    stm.str("");
    stm<<"../"<<input_name<<n_neigh<<"_result.csv";
    return stm.str();
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

//bool isPointOutside()
//{
//    app.setPoint(points[i]);
//    float moy_proj1 = 0;
//    float moy_proj2 = 0;
//    float sum_poids = 0;
//    for(int c = 0; c< neighborhood_.size(); ++c)
//    {
//        if(poids_[c]> thresh_weigh)
//        {
//            moy_proj1 += poids_[c] * normal.dot(dist_[c]);
//            sum_poids += poids_[c] ;
//        }

//        moy_proj1 /= sum_poids;

//        sum_poids = 0;

//        if(poids_[c]< 0.1)
//        {
//            moy_proj2 += normal.dot(dist_[c]);
//            ++sum_poids;
//        }

//        moy_proj2 /= sum_poids;
//    }



//    if(moy_proj1>-0.005 || moy_proj2>-0.005)
//        return false;

//    return true;
//}
