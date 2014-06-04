#pragma once

#include "boost\filesystem.hpp"

namespace caffe
{

	using namespace boost::filesystem;

	// recursive make directory
	void mkdir(const std::string& path);

	// recursive find all files with given extension
	// E.G, if you want to find all jpg and png files, ext_filter = "jpg|png";
	void get_all_files(std::string root, std::vector<std::string>& files, std::string ext_filter="*");
};