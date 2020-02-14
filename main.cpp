// -------------------------------------------------------------------------------------
//        Iterative weighted PCA for robust and edge-aware normal vector estimation
//--------------------------------------------------------------------------------------
// Julia Sanchez, Florence Denis, David Coeurjolly, Florent dupont, Laurent Trassoudaine, Paul Checchin
// Liris (Lyon), Institut Pascal (Clermont Ferrand)
// Région Auvergne Rhône Alpes ARC6
// Please refer to linked paper if using this code
// --------------------------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <omp.h>

#include "app.h"
#include "cloud.h"

void initFile(std::string );
void write(std::string , std::vector<Eigen::Vector3f> , std::vector<Eigen::Vector3f> );
std::string create_output_name(std::string, std::string, int);

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cout<<std::endl;
        printf("./normal_estimation [pointcloud] [number of neighbors] [noise] [curvature] [radius_or_knn]\n");
        printf("Estimated noise (m),\n \t example : 0.0001 \n");
        printf("Min tolerated curvature radius (m) \n \t example for curved objects = 0.1 \n \t for intersecting planes = 10000000000\n");
        printf("Neighborhood defined as radius (0) or knn (1)");
        printf("It is recommended to scale the objects (for diagonal to be 1) to better parameterize the min tolerated curvature radius");
        return 0;
    }
    else
    {
        std::cout<<std::endl;
        for (int i = 1 ; i<argc; i++)
            std::cout<<argv[i]<<"   ";
        std::cout<<std::endl<<std::endl;
    }

    //------------------------------------------------------------------------------------------------

    cloud c;
    std::string input_name = argv[1];
    int cloud_size = c.Read(input_name);
    c.buildTree();
    float res = c.getResolution();
    std::cout<<"cloud resolution : "<<res<<std::endl<<std::endl;

    Eigen::Matrix<float, Eigen::Dynamic, 3>* pc = c.getPC();
    flann::Index<flann::L2<float>>* tree = c.getTree();

    int n_neigh;
    float radius;
    bool knn_or_radius = atoi(argv[5]);

    //Select knn or radius search
    if(knn_or_radius)
    	n_neigh = atoi(argv[2]);
    else
    	radius = atof(argv[2]);

    int n = 0;
    int nan = 0;
    float noise = (float)atof(argv[3])/sqrt(3);
    noise = std::max(noise, noise_min);
    float curvature = atof(argv[4]);

    std::cout<<"estimated noise="<<noise<<std::endl<<std::endl;
    std::cout<<"minimum curvature radius tolerated="<<curvature<<std::endl<<std::endl;
    std::cout<<"division factor="<<div_fact<<std::endl<<std::endl;

    std::vector<Eigen::Vector3f> normals(cloud_size);
    std::vector<Eigen::Vector3f> points(cloud_size);
    std::vector<int> onEdge(cloud_size);

    std::cout<<"------------------------------Computing normals------------------------------"<<std::endl<<std::endl;
    auto t_tot1 = std::chrono::high_resolution_clock::now();


/// ----------------------------------------------------MAIN LOOP------------------------------------------------------------------------------------

    std::cout<<"Avancement : "<<std::endl;

    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) shared(onEdge, pc, tree, noise, n_neigh, curvature, normals, points, radius, knn_or_radius) // comment if one thread used
    for (int i = 0; i < cloud_size; ++i)
    {
        CApp app(pc, tree, i, noise);
        Eigen::Vector3f point_ref = app.getPoint();
        app.setParams(div_fact, curvature);
	if(knn_or_radius)
        	app.selectNeighborsKnn(n_neigh);
	else
		app.selectNeighborsRadius(radius);

        if(!(i%1000))
            std::cout<<((float)i/(float)cloud_size) * 100<<"%"<<std::endl;
        app.init1();

        if( app.isOnEdge() ) // when the mean projection error is greater than the noise-------------------------------------------------------------------------------------------------------
        {
            onEdge[i] = 1;

            //Compute first solution n_1------------------------------------------------------------------------------------------------------
            bool first = true;
            app.Optimize(first);
            app.OptimizePos(first, thresh_weight);

            //Compute second solution n_2------------------------------------------------------------------------------------------------------
            app.reinitPoint();
            app.init2();
            first = false;
            if(app.SuspectedOnEdge_)
            {
                app.Optimize(first);
                app.OptimizePos(first, thresh_weight);
            }

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
            onEdge[i] = 0;

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

        points[i] = point_ref;

        //orient through exterior (relatively to viewpoint)
        if(normals[i].dot(points[i])>0)
            normals[i] *= -1;
    }
    auto t_tot2 = std::chrono::high_resolution_clock::now();

    //save normals
    std::string output = create_output_name(input_name, "mymet", n_neigh);
    write(output, points, normals);

    //save time
    std::string time_name = create_output_name(input_name, "mymet_time", n_neigh);
    std::ofstream ftime(time_name, std::ofstream::trunc);
    ftime<<std::chrono::duration_cast<std::chrono::milliseconds>(t_tot2-t_tot1).count();
    ftime.close();

    return 0;
}

//____________________________________________________________- OUT FUNCTIONS -___________________________________________________________


std::string create_output_name(std::string input_name, std::string what, int n_neigh)
{
    size_t lastindex_point = input_name.find_last_of(".");
    size_t lastindex_slash = input_name.find_last_of("/");
    if (lastindex_slash==std::string::npos)
       lastindex_slash = -1;

    input_name = input_name.substr(lastindex_slash+1, lastindex_point-(lastindex_slash+1));
    std::stringstream stm;
    stm.str("");
    stm<<input_name<<"_"<<n_neigh<<"_"<<what<<".csv";
    return stm.str();
}

void initFile(std::string file_name)
{
    std::ofstream fout;
    fout.open(file_name, std::ofstream::trunc);
    if (fout.is_open())
        fout<<"x,y,z,nx,ny,nz\n";
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
                fout<<points[i](j)<<",";
            for (int j = 0; j<2; ++j)
            {
                if(abs(normals[i](j))>1e-30)
                    fout<<normals[i](j)<<",";
                else
                    fout<<"0,";
            }
            if(abs(normals[i](2))>1e-30)
                fout<<normals[i](2)<<"\n";
            else
                fout<<"0\n";
        }

        fout.close();
    }
}
