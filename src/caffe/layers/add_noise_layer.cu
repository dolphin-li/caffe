#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	Dtype AddNoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		for(size_t i = 0; i<bottom.size(); i++)
		{
			// generate noise-levels
			generate_noise_levels();
			const Dtype* noise_level_data = noise_levels_.gpu_data();
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = (*top)[i]->mutable_gpu_data();
			Dtype* noise_data = noises_.mutable_gpu_data();

			const int n = bottom[i]->num();
			const int np = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
			caffe_gpu_rng_gaussian(n*np, Dtype(0), Dtype(1), noise_data);
			caffe_gpu_dgmm(CUBLAS_SIDE_RIGHT, np, n, noise_data, noise_level_data, noise_data);
			caffe_gpu_add(n*np, bottom_data, noise_data, top_data);
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(AddNoiseLayer);

}  // namespace caffe
