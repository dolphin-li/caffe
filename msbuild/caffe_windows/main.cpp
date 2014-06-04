// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include "caffe/caffe.hpp"
#include "caffe\util\folderhelper.h"
#include <Windows.h>

using namespace caffe;
using namespace boost::filesystem;


int train(int argc, char* argv[])
{
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);
	string logPreFix = std::string("./logs/") + path(argv[1]).filename().string() + "/";
	mkdir(logPreFix);
	::google::SetLogDestination(0, logPreFix.c_str());
	if (argc < 2) 
	{
		LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
		return 0;
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

	return 1;
}


int main(int argc, char* argv[]) 
{

	train(argc, argv);

	return 0;
}
