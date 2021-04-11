//
// Created by "Dylan Brasseur" on 06/02/2020.
//
#include <algorithm>
#include <random>
#include <fstream>
#include "../include/Category.h"
#include "../include/computeFunctions.h"

std::mutex Category::disks_access;

void Category::setTargetDisks(std::vector<Disk> const &target){
    target_disks = target;
}

void Category::addDependency(unsigned long parent_id){
    if(std::find(parents_id.begin(), parents_id.end(), parent_id) == parents_id.end())
    {
        parents_id.push_back(parent_id);
    }
}

void Category::addChild(unsigned long child_id){
    if(std::find(children_id.begin(), children_id.end(), child_id) == children_id.end())
    {
        children_id.push_back(child_id);
    }
}

void Category::computeTarget(){
    target_pcf.clear();
    target_rmax.clear();
    target_radii.clear();

    if(target_disks.empty())
    {
        return;
    }

    auto nSteps = (unsigned long)(params->limit/params->step);

    target_rmax.insert(std::make_pair(id, computeRmax(target_disks.size())));
    for(unsigned long parent : parents_id)
    {
        // std::cout << "id: " << id << ", parent: " << parent << std::endl;
        target_rmax.insert(std::make_pair(parent, computeRmax(target_disks.size()/*+(*categories.get())[parent].target_disks.size()*/)));
    }
    std::vector<float> area, radii;
    area.resize(nSteps);
    radii.resize(nSteps);
    std::vector<unsigned long> relations;
    relations.push_back(id);    // build relations, including itself and all parents
    relations.insert(relations.end(), parents_id.begin(), parents_id.end());

    for(unsigned long parent : relations)
    {   
        // std::cout << "id: " <<id << ", parent: " << parent << std::endl;
        // if (id != 0 || parent != 0) {     // TODO: remove this
        //     continue;   
        // }
        float rmax = target_rmax[parent];
        std::cout << "id: " <<id << ", parent: " << parent << " " << rmax << std::endl;
        for(unsigned long i=0; i<nSteps; i++)
        {
            float r = (i+1)*params->step;
            float outer = (r+0.5f)*rmax;
            float inner = std::max((r-0.5f)*rmax, 0.f);
            area[i] = M_PI*(outer*outer - inner*inner);
            radii[i] = r*rmax;
        }
        target_areas.insert(std::make_pair(parent, area));
        target_radii.insert(std::make_pair(parent, radii));

        auto & parent_disks = (*categories.get())[parent].target_disks;
        target_pcf.insert(std::make_pair(parent, compute_pcf(target_disks, parent_disks, area, radii, rmax, *params.get())));
    }
}



void Category::initialize(float domainLength, float e_delta){
    if(initialized)
        return;
    std::random_device rand_device;
    std::mt19937_64 rand_gen(rand_device());

    disks.clear();
    pcf.clear();

    //Initialize the parents before this one (akin to the topological order)
    for(unsigned long parent : parents_id)
    {
        std::cout << "topological order: " << id << " " << parent << std::endl;
        (*categories.get())[parent].initialize(domainLength, e_delta);
    }

    // std::cout << "id: " << id << std::endl;

    std::vector<float> output_disks_radii;

    //Adapt to the domain length
    float n_factor = domainLength*domainLength;
    float diskfact = 1/domainLength;
    unsigned long long n_repeat = std::ceil(n_factor);
    output_disks_radii.reserve(n_repeat*target_disks.size());
    for(auto & d : target_disks)
    {
        for(unsigned long long i=0; i<n_repeat; i++)
        {
            output_disks_radii.push_back(d.r);
        }
    }
    std::uniform_real_distribution<float> randf(0, domainLength);

    std::shuffle(output_disks_radii.begin(), output_disks_radii.end(), rand_gen); //Shuffle array
    output_disks_radii.resize(target_disks.size()*n_factor); // and resize it to the number of disks we want
    // This combination effectively does a random non repeating sampling

    std::sort(output_disks_radii.rbegin(), output_disks_radii.rend()); //Sort the radii in descending order
    finalSize = output_disks_radii.size();

    // std::cout << id << " " << output_disks_radii.size() << std::endl;
    float e_0 = 0;
    unsigned long max_fails=1000;
    unsigned long fails=0;
    unsigned long n_accepted=0;
    // std::cout << disks.size() << " " << output_disks_radii.size() << std::endl;
    disks.reserve(output_disks_radii.size());
    
    auto nSteps = (unsigned long)(params->limit/params->step);
    std::vector<unsigned long> relations;
    relations.reserve(1+parents_id.size());
    relations.push_back(id);
    relations.insert(relations.end(), parents_id.begin(), parents_id.end());
    auto & others = *categories.get();
    auto & parameters = *params.get();

    constexpr unsigned long MAX_LONG = std::numeric_limits<unsigned long>::max();
    std::map<unsigned long, std::vector<std::vector<float>>> weights;
    std::map<unsigned long, std::vector<float>> current_pcf;

    // Compute the weights for each realtion disks
    for(auto relation : relations){

        // if (id != 0 || relation != 0) {
        //     return;
        // }
        // std::cout << id << " " << relation << std::endl;
        current_pcf.insert(std::make_pair(relation, 0));
        current_pcf[relation].resize(nSteps, 0);

        // std::cout << others[relation].disks.size() << std::endl;
        std::vector<std::vector<float>> current_weight = get_weights(others[relation].disks, target_radii[relation], diskfact);
        // std::cout << "current_weight.size(): " << current_weight.size() << std::endl;
        // for (int i=0; i<current_weight.size(); i++) {
        //     for (int j=0; j<current_weight[i].size(); j++) {
        //         std::cout << current_weight[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        weights.insert(std::make_pair(relation, current_weight));
    }

    // original code
    // for(auto relation : relations){
    //     std::cout << "current_weight.size(): " << others[relation].disks.size() << std::endl;
    //     current_pcf.insert(std::make_pair(relation, 0));
    //     current_pcf[relation].resize(nSteps, 0);
    //     weights.insert(std::make_pair(relation, get_weights(others[relation].disks, target_radii[relation], diskfact)));
    // }

    std::map<unsigned long, Contribution> contributions;

    // std::ofstream out_debug("../outputs/debug_forest_"+std::to_string(id)+".txt");

    std::uniform_real_distribution<float> rand_0to1(0, 1);
    
    do{
        // std::cout << "Error: " << id <<std::endl;
        bool rejected=false;
        float e = e_0 + e_delta*fails;
        // std::cout << "e: " << n_accepted << " " << e << std::endl;
        //Generate a random disk

        float rx = randf(rand_gen);
        float ry = randf(rand_gen);

        // float rx = rand_0to1(rand_gen);
        // float ry = rand_0to1(rand_gen);
        float min_xy = 0.09;
        if (id == 0) {
            rx = min_xy + rx * (domainLength - min_xy * 2);
            ry = min_xy + ry * (domainLength - min_xy * 2);
        } 
        
        Disk d_test(rx, ry, output_disks_radii[n_accepted]);
        
        // std::cout << "n_accepted: " << n_accepted << std::endl;
        for(auto relation : relations)
        {
            Contribution test_pcf;
            if(!disks.empty() || relation != id)
            {
                //Computing the contribution of this disk to the pcf for this relation
                
                test_pcf = compute_contribution(d_test, others[relation].disks, weights[relation], target_radii[relation], target_areas[relation], target_rmax[relation], parameters, relation == id ? n_accepted : MAX_LONG,relation == id ? 2*output_disks_radii.size()*output_disks_radii.size() : 2*output_disks_radii.size()*others[relation].disks.size(), diskfact);
                
                // std::cout << "disks: " << disks.size() << std::endl;     // both increasing
                // std::cout << "others: " << others[relation].disks.size() << std::endl;
                float ce = compute_error(test_pcf, current_pcf[relation], target_pcf[relation]);
                // if (id == 1 && relation == 0) {
                //     std::cout << "n_accepted: " << n_accepted << std::endl;
                //     for (int kk=0; kk<test_pcf.contribution.size(); kk++) {
                //         std::cout << test_pcf.contribution[kk] << " ";
                //     }
                //     std::cout << std::endl;
                // }

                if(e < ce) {
                    //Disk is rejected if the error is too high
                    rejected=true;
                    break;
                } else {

                }
            }else{

                test_pcf.pcf.resize(nSteps, 0);
                test_pcf.contribution.resize(nSteps, 0);
                test_pcf.weights = get_weight(d_test, target_radii[relation], diskfact);
            }
            contributions.insert_or_assign(relation, test_pcf);
        }
        if(rejected)
        {
            fails++;
        }else
        {
            //The disk is accepted, we add it to the list
//            disks_access.lock();
            std::cout << "===> Before Grid Search, n_accepted: " << n_accepted+1 << "/" << output_disks_radii.size() << std::endl;
            disks.push_back(d_test);
//            disks_access.unlock();
            fails=0;
            // if (id == 0) {
            // out_debug << rx << " " << ry << " " << output_disks_radii[n_accepted] << std::endl;
            // std::cout << rx << " " << ry << " " << output_disks_radii[n_accepted] << std::endl;
            // }
            // if (id == 1) {
            //     std::cout << n_accepted << " " << rx << " " << ry << " " << output_disks_radii[n_accepted] << std::endl;
            // }
            for(auto relation : relations)
            {
                auto & current = current_pcf[relation];
                auto & contrib = contributions[relation];
                if(relation == id)
                {
                    weights[relation].emplace_back(contrib.weights);
                }
                for(unsigned long k=0; k<nSteps; k++)
                {   
                    
                    current[k]+=contrib.contribution[k];
                }

                // newly added for debugging
                // if (id == 1) {
                //     for(unsigned long k=0; k<nSteps; k++) {   
                //         std::cout << current[k] << " ";
                //     }
                // }
                // std::cout << std::endl;
            }
            n_accepted++;
        }

        if(fails > max_fails)
        {
            //We have exceeded the 1000 fails threshold, we switch to a parallel grid search
            
            //Grid search
            constexpr unsigned long N_I = 80*2;
            constexpr unsigned long N_J = 80*2;
            std::map<unsigned long, Contribution> contribs[N_I][N_J];
            while(n_accepted < output_disks_radii.size())
            {
                std::cout << "===> Doing Grid Search: " << n_accepted+1 << "/" << output_disks_radii.size() <<std::endl;
                float errors[N_I+1][N_J+1];
                Compare minError = {INFINITY,0, 0};
// #pragma omp parallel for default(none) collapse(2) shared(output_disks_radii, n_accepted, relations, others, parameters, nSteps, errors, diskfact, contribs, weights, current_pcf, domainLength)
                for(unsigned long i=1; i<N_I; i++)
                {
                    for(unsigned long j=1; j<N_J; j++)
                    {
                        float currentError=0;
                        float cell_x = (domainLength/N_I)*i;
                        float cell_y = (domainLength/N_J)*j;
                        Disk cell_test(cell_x, cell_y, output_disks_radii[n_accepted]);
                        // // std::cout << cell_x << " " << cell_y << std::endl;
                        // if (id == 0) {
                        //     if ((cell_x < min_xy || cell_x > domainLength - min_xy) || (cell_y < min_xy || cell_y > domainLength - min_xy)) {
                        //         // std::cout << cell_x << " " << cell_y << std::endl;
                        //         continue;
                        //     } else {
                        //         // std::cout << cell_x << " " << cell_y << std::endl;
                        //     }
                        // }
                        

                        for(auto && relation : relations)
                        {
                            Contribution test_pcf;
                            test_pcf = compute_contribution(cell_test, others[relation].disks, weights[relation], target_radii[relation], target_areas[relation], target_rmax[relation], parameters, relation == id ? n_accepted : MAX_LONG, relation == id ? output_disks_radii.size()*output_disks_radii.size() : output_disks_radii.size()*others[relation].disks.size(), diskfact);
                            float ce = compute_error(test_pcf, current_pcf[relation], target_pcf[relation]);
                            currentError = std::max(currentError, ce);
                            // std::cout << "ce: " << ce << std::endl;
                            contribs[i][j].insert_or_assign(relation, test_pcf);
                        }
                        
                        errors[i][j] = currentError;
                        // newly modified
                        if(errors[i][j] < minError.val)
                        {
                            minError.val = errors[i][j];
                            minError.i = i;
                            minError.j = j;
                        }
                    }
                }

                // for(unsigned long i=1; i<N_I; i++)
                // {
                //     for(unsigned long j=1; j<N_J; j++)
                //     {
                //         if(errors[i][j] < minError.val)
                //         {
                //             minError.val = errors[i][j];
                //             minError.i = i;
                //             minError.j = j;
                //         }
                //     }
                // }


                //We automatically accept the disk with the lowest error
//                disks_access.lock();
                
                disks.emplace_back((domainLength/N_I)*minError.i + (randf(rand_gen)-domainLength/2)/(N_I*10), (domainLength/N_J)*minError.j + (randf(rand_gen)-domainLength/2)/(N_J*10), output_disks_radii[n_accepted]);
                
                
//                disks_access.unlock();
                for(auto relation : relations)
                {
                    auto & current = current_pcf[relation];
                    auto & contrib = contribs[minError.i][minError.j][relation];
                    if(relation == id)
                    {
                        weights[relation].emplace_back(contrib.weights);
                    }
                    for(unsigned long k=0; k<nSteps; k++)
                    {
                        current[k]+=contrib.contribution[k];
                    }
                }
                n_accepted++;
                // std::cout << "n_accepted: " << n_accepted << std::endl;
            }

        }
    }while(n_accepted < output_disks_radii.size());
//    for(auto r : relations)
//    {
//        //We're done with the initialisation, we recompute a pcf for the whole class to eliminate round off errors and such
//        pcf.insert_or_assign(r, compute_pcf(disks, others[r].disks, target_areas[r], target_radii[r], target_rmax[r], parameters));
//    }
    initialized=true;
    // out_debug.close();

}

std::vector<Target_pcf_type> Category::getCurrentPCF(unsigned long parent){
    return pcf.at(parent);
}

std::vector<Target_pcf_type> Category::getTargetPCF(unsigned long parent){
    return target_pcf.at(parent);
}

std::vector<std::pair<std::pair<unsigned long, unsigned long>, std::vector<std::pair<float, float>>>> Category::getCurrentPCFs(){
    std::vector<std::pair<std::pair<unsigned long, unsigned long>, std::vector<std::pair<float, float>>>> result;
    result.reserve(pcf.size());
    for(auto & currpcf : pcf)
    {
        result.emplace_back(std::make_pair(currpcf.first, id), 0);
        auto & coords = result.back().second;
        coords.reserve(currpcf.second.size());
        for(auto & value : currpcf.second)
        {
            coords.emplace_back(value.radius, value.mean);
        }
    }

    return result;
}

std::vector<Disk> Category::getCurrentDisks(){
    disks_access.lock();
    std::vector<Disk> outDisks(disks);
    disks_access.unlock();
    return outDisks;
}

void Category::addTargetDisk(Disk const &d){
    target_disks.push_back(d);
}

std::vector<Disk> Category::getTargetDisks(){
    return target_disks;
}

std::vector<std::pair<std::pair<unsigned long, unsigned long>, std::vector<std::pair<float, float>>>> Category::getTargetPCFs(){
    std::vector<std::pair<std::pair<unsigned long, unsigned long>, std::vector<std::pair<float, float>>>> result;
    result.reserve(target_pcf.size());
    for(auto & currpcf : target_pcf)
    {
        result.emplace_back(std::make_pair(currpcf.first, id), 0);
        auto & coords = result.back().second;
        coords.reserve(currpcf.second.size());
        for(auto & value : currpcf.second)
        {
            coords.emplace_back(value.radius, value.mean);
        }
    }

    return result;
}

Compute_status Category::getComputeStatus()
{
    Compute_status status;
    status.rmax = target_rmax[id];
    status.disks = getCurrentDisks();
    status.parents = parents_id;
    return status;
}

Compute_status Category::getTargetComputeStatus()
{
    Compute_status status;
    status.rmax = target_rmax[id];
    status.disks = getTargetDisks();
    status.parents = parents_id;
    return status;
}

unsigned long Category::getFinalSize(float domainLength){
    return target_disks.size()*domainLength*domainLength;
}

void Category::normalize(float domainLength)
{
    for(auto & d : disks)
    {
        d.x/=domainLength;
        d.y/=domainLength;
        d.r/=domainLength;
    }
}

void Category::refine(unsigned long max_iter, float threshold, bool isDistanceThreshold){
    //TODO
}
