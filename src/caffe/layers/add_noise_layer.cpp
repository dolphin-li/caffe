#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void AddNoiseLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		CHECK_GE(bottom.size(), 1);		  
		CHECK_EQ(bottom.size(), top->size());
		CHECK_EQ(bottom[0]->channels(), 3) << "only BGR-3 channels accepted";
		CHECK_EQ((*top)[0]->channels(), 1) << "only 1-channel output accepted";
		CHECK_EQ(layer_param_.add_noise_param().noise_level_first_sample_point_size(),
			layer_param_.add_noise_param().noise_level_first_sample_rate_size());
		CHECK_EQ(layer_param_.add_noise_param().noise_type(), 0) << "noise types: 0=Gaussian (only 0 allowed now)";

		for(size_t i=0; i<bottom.size(); i++)
			(*top)[i]->ReshapeLike(*bottom[i]);
		noise_levels_.Reshape(bottom[0]->num(), 1, 1, 1);
		noises_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void AddNoiseLayer<Dtype>::generate_noise_levels()
	{
		const float *nl_emphs = layer_param_.add_noise_param().noise_level_first_sample_point().data();
		const float *nl_emphs_rates = layer_param_.add_noise_param().noise_level_first_sample_rate().data();
		int n_emphs = layer_param_.add_noise_param().noise_level_first_sample_point_size();

		float nlMin = layer_param_.add_noise_param().noise_level_min();
		float nlMax = layer_param_.add_noise_param().noise_level_max();
		Dtype* data = noise_levels_.mutable_cpu_data();
		caffe_rng_uniform(noise_levels_.count(), (Dtype)nlMin, (Dtype)nlMax, data);

		std::vector<int> emphsNums(n_emphs+1, 0);
		for(int i=0; i<n_emphs; i++)
		{
			int num = int(nl_emphs_rates[i]*noise_levels_.count());
			emphsNums[i+1] = emphsNums[i] + num;
			CHECK_LE(num, noise_levels_.count()) << "number of emphasized data must < total data";
		}

		for(int i=0; i<n_emphs; i++)
		{
			int b = emphsNums[i];
			int e = emphsNums[i+1];
			for(int j=b; j<e; j++)
				data[j] = nl_emphs[j]; 
		}
	}

	template <typename Dtype>
	Dtype AddNoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		for(size_t i = 0; i<bottom.size(); i++)
		{
			// generate noise-levels
			generate_noise_levels();
			const Dtype* noise_level_data = noise_levels_.cpu_data();
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = (*top)[i]->mutable_cpu_data();
			Dtype* noise_data = noises_.mutable_cpu_data();

			const int n = bottom[i]->num();
			const int np = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
			caffe_rng_gaussian(n*np, Dtype(0), Dtype(1), noise_data);
			caffe_cpu_dgmm(CUBLAS_SIDE_RIGHT, np, n, noise_data, noise_level_data, noise_data);
			caffe_add(n*np, bottom_data, noise_data, top_data);
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(AddNoiseLayer);

}  // namespace caffe
