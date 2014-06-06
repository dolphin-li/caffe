// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include "caffe/caffe.hpp"
#include "caffe\util\folderhelper.h"
#include <Windows.h>
#include <random>

using namespace caffe;
using namespace boost::filesystem;
using std::endl;
using std::cout;


int train(int argc, char* argv[])
{
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);
	string path,name,ext;
	path_name_ext(argv[1], path, name, ext);
	string logPreFix = std::string("./logs/") + name + "/";
	mkdir(logPreFix);
	::google::SetLogDestination(0, logPreFix.c_str());
	if (argc < 2) 
	{
		LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
		return -1;
	}
	SolverParameter solver_param;
	ReadProtoFromTextFile(argv[1], &solver_param);

	LOG(INFO) << "Starting Optimization"; 
	SGDSolver<float> solver(solver_param);
	if (argc == 3) {
		LOG(INFO) << "Resuming from " << argv[2];
		solver.Solve(argv[2]); 
	} else {
		solver.Solve(); 
	}
	LOG(INFO) << "Optimization Done.";

	return 0;
}


int generate_image_list(int argc, char* argv[])
{
	if(argc < 2)
	{
		cout << "usage: " << argv[0] << " root_dir [file_extension, e.g., .jpg]" << endl;
	}
	std::string root = argv[1];
	std::string ext = ".*";
	if(argc >= 3)
	{
		ext = argv[2];
	}

	std::vector<std::string> imglist;
	get_all_files(root, imglist, ext);
	string filename = root+".imagelist.txt";
	std::ofstream file(filename);
	if(file.fail())
	{
		cout << "open failed: " << filename << endl;
		return -1;
	}

	for(size_t i=0; i<imglist.size(); i++)
	{
		file << imglist[i] << " " << 0 << endl;
	}
	file.close();
}

int main(int argc, char* argv[]) 
{
	return generate_image_list(argc, argv);

	return train(argc, argv);
	
	return 0;
}
