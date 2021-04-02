#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <thread>
#include <fstream>
#include <sstream>
#include <cassert>
#include "include/ASMCDD.h"

// unsigned long duration;
std::string TOTAL_SIZE;
ASMCDD algo;
ASMCDD_params algo_params;
std::vector<unsigned long> currentSizes;
std::vector<unsigned long> finalSizes;
unsigned long totalSize;


void parse_example(std::string const &fileName){
    std::cout << "Loading " << fileName << std::endl;
    std::ifstream file(fileName);
    assert(file.good());
    std::string path;
    // unsigned int r, g, b, id_a, id_b, count;
    unsigned int id_a, id_b, count;
    std::getline(file, path); //File with example
    algo.loadFile(path);
    // std::getline(file, path);
    // count = std::stoi(path); // Number of classes
    // char text[32] = "";
    // for(unsigned int i = 0; i < count; i++) //Add mesh and set color for each class
    // {
    //     std::getline(file, path);
    //     std::getline(file, path);
    //     std::stringstream line(path);
    //     line >> r >> g >> b;
    // }
    std::getline(file, path);
    count = std::stoi(path); // Number of dependecies
    for(unsigned int i = 0; i < count; i++) // Add dependencies
    {
        std::getline(file, path);
        std::stringstream line(path);
        line >> id_a >> id_b;
        algo.addDependency(id_a, id_b);
    }
    // while(std::getline(file, path)){
    //     std::stringstream line(path);
    //     line >> id_a >> id_b >> r >> g >> b;
    //     std::sprintf(text, "%u %u", id_a, id_b);
    // }
    // file.close();

    finalSizes = algo.getFinalSizes(algo_params.domainLength);
    currentSizes.resize(finalSizes.size(), 0);
    totalSize = std::accumulate(finalSizes.begin(), finalSizes.end(), 0UL);
    TOTAL_SIZE = std::to_string(totalSize);
    algo.setParams(algo_params);
}


void parse_arguments(int argc, char **argv){
    if(argc < 2){
        std::cerr << "Usage : " << argv[0]
                  << " example_config_file [domain_length [error_delta [sigma [step [limit [max_iter [threshold isDistance] ]]]]]]"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }else{
        switch(argc){
            case 9 :
                std::cerr
                        << "If threshold is given, you need to say if it's a distance threshold in the next argument with a positive integer"
                        << std::endl;
                std::exit(EXIT_FAILURE);
            default:
            case 10:
                algo_params.distanceThreshold = std::stoi(argv[9]) > 0;
                algo_params.threshold = std::stof(argv[8]);
            case 8:
                algo_params.max_iter = std::stoul(argv[7]);
            case 7:
                algo_params.limit = std::stof(argv[6]);
            case 6:
                algo_params.step = std::stof(argv[5]);
            case 5:
                algo_params.sigma = std::stof(argv[4]);
            case 4:
                algo_params.error_delta = std::stof(argv[3]);
            case 3:
                algo_params.domainLength = std::stof(argv[2]);
            case 2:
                algo_params.example_filename = argv[1];
        }
    }
}


int main(int argc, char *argv[]){

    // parse input
    parse_arguments(argc, argv);
    parse_example(algo_params.example_filename);
    std::string example_filename = algo_params.example_filename;
    // find last /
    std::size_t found = example_filename.find_last_of("/");  // tested on macOS
    example_filename = example_filename.substr(found+1);
    // find .
    found = example_filename.find(".");  // tested on macOS
    example_filename = example_filename.substr(0, found);
    std::cout << "===> Running Scene: " << example_filename << std::endl;

    // run algo
    algo.computeTarget();
    
    auto plots = algo.getPrettyTargetPCFplot(1);
    
    //Initialization
    std::vector<unsigned long> vs = algo.getFinalSizes(1);
    std::cout << "Total number of class: " << vs.size() << std::endl;
    algo.initialize(algo_params.domainLength, algo_params.error_delta);
    
    // write output to file
    std::ofstream out_init_pts("../outputs/"+example_filename+"_init.txt");
    out_init_pts << vs.size();
    for (unsigned long id=0; id<vs.size(); id++) {
        out_init_pts << " " << (id + 1) * 1000;
    }
    out_init_pts << std::endl;
    for (unsigned long id=0; id<vs.size(); id++) {
        std::vector<Disk> ds = algo.getCurrentDisks(id);
        for (int j=0; j<ds.size(); j++) {
            float x = ds[j].x * 10000;
            float y = ds[j].y * 10000;
            float r = ds[j].r * 10000;
            out_init_pts << (id + 1) * 1000 << " " << (int)x << " " << (int)y << " " << (int)r << std::endl;
        }
    }
    out_init_pts.close();
    std::cout << "Done !" << std::endl;

}
