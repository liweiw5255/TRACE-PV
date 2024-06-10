#include "../include/a2s_header.h"
#include "../include/a2s_parameters.h"

extern size_t getHash(std::vector<double> vec){
    std::hash<std::string> key;
    std::string str(vec.begin(),vec.end());
    //get the hash value of current string
    return key(str);
}

extern std::unordered_map<size_t, std::vector<size_t> > clusterMerging(std::vector<std::vector<double> > MP_Input, size_t sz){
  
    //unordered_map<size_t, std::vector<double> > dict;    
    std::vector<double> min(MP_sz,0), max(MP_sz,0), thr(MP_sz,0);

    for(size_t i=0; i<MP_sz; ++i){
        for(size_t j=0; j<sz; ++j){
            if(MP_Input[j][i]<min[i])
                min[i]=MP_Input[j][i];
            if(MP_Input[j][i]>max[i])
                max[i]=MP_Input[j][i];
            thr[i]+=MP_Input[j][i];
        }
        thr[i]/=sz;
        thr[i]*=threshold[i];
    }

    int category;
    std::vector<double> tmp; 

    for(size_t i=0; i<MP_sz; ++i){
        for(size_t j=0; j<sz; ++j){
            category=(MP_Input[j][i]-min[i])/threshold[i];            
            tmp.push_back(category*threshold[i]+min[i]);
        }
    } 

    std::vector<double> vec;
    std::vector<size_t> key_list;
    std::unordered_map<size_t, std::vector<size_t> > dict;    
    
    for(size_t i=0; i<sz; ++i){
        vec=MP_Input[i];
        size_t hash_value=getHash(vec);
        key_list.push_back(hash_value);
        dict[hash_value].push_back(i);
    }

    // Traverse
    //unordered_map<size_t, int>::iterator iter;
    //for(iter=dict.begin();iter!=dict.end();++iter)
    //    cout<<iter->first<<ends<<iter->second<<endl;

    //cout<<dict.size()<<endl;
    return dict;

}