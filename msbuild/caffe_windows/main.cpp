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
#include "boost\filesystem.hpp"
#include <Windows.h>

using namespace caffe;
using namespace boost::filesystem;


std::string unicode2ascii(const wchar_t* src)
{
	int nSrcLen = lstrlenW( src ); // Convert all UNICODE characters
	int nDstLen = WideCharToMultiByte( CP_ACP, // ANSI Code Page
		0, // No special handling of unmapped chars
		src, // wide-character string to be converted
		nSrcLen,
		NULL, 0, // No output buffer since we are calculating length
		NULL, NULL ); // Unrepresented char replacement - Use Default 

	std::vector<char> dst;
	dst.resize(nDstLen);

	WideCharToMultiByte( CP_ACP, // ANSI Code Page
		0, // No special handling of unmapped chars
		src, // wide-character string to be converted
		nSrcLen,
		dst.data(), 
		nDstLen,
		NULL, NULL ); // Unrepresented char replacement - Use Default
	return dst.data();
}

int train(int argc, char* argv[])
{
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);

	create_directory("./logs/");

	path path1(argv[1]);

	string logPreFix = std::string("./logs/") + unicode2ascii(path1.filename().c_str());
	::google::SetLogDestination(0, logPreFix.c_str());
	if (argc < 2) {
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
}


int main(int argc, char* argv[]) 
{

	train(argc, argv);

	return 0;
}
