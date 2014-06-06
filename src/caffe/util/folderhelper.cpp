#include "caffe/util/folderhelper.h"
#include "boost/regex.hpp"
namespace caffe
{
	// recursive make directory
	void mkdir(const std::string& filename)
	{
		boost::filesystem::path path1(filename);
		path1 = path1.remove_filename();
		std::vector<path> dirs2make;
		for(path tp = path1; !exists(tp) && !tp.empty(); tp = tp.parent_path())
			dirs2make.push_back(tp);
		for(std::vector<path>::reverse_iterator it = dirs2make.rbegin(); it != dirs2make.rend(); ++it)
			boost::filesystem::create_directory(*it);
	}

	std::string fullfile(const std::string& path, const std::string& name)
	{
		std::string s = path;
		if(s.size() == 0)
			return name;
		if(s.back() != '/' && s.back() != '\\')
			s += '/';
		s += name;
		return s;
	}

	void path_name_ext(const std::string& full, std::string& path, std::string& name, std::string& ext)
	{
		boost::filesystem::path path1(full);

		ext = path1.extension().string();
		path = path1.remove_filename().string();
		name = path1.filename().string();
		name = name.substr(0, name.find_last_of('.'));
	}

	// recursive find all files with given extension
	void get_all_files(std::string root, std::vector<std::string>& files, std::string ext)
	{
		if ( !exists( root ) ) return;

		const boost::regex my_filter( ext );

		directory_iterator end_itr; // default construction yields past-the-end
		for ( directory_iterator itr( root ); itr != end_itr; ++itr )
		{
			if ( is_directory(itr->status()) )
			{
				get_all_files( itr->path().string(), files, ext );
			}
			else if ( boost::regex_match( itr->path().extension().string(), my_filter ) ) // see below
			{
				files.push_back(itr->path().string());
			}
		}
	}
};