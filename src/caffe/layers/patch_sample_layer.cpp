#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void PatchSampleLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		CHECK_GE(bottom.size(), 1);		  
		CHECK_EQ(bottom.size(), top->size());

		const int n_per_img = layer_param_.patch_sample_param().sample_per_img();
		const int sz = layer_param_.patch_sample_param().patch_size();

		for(size_t i=0; i<bottom.size(); i++)
			(*top)[i]->Reshape(bottom[i]->num()*n_per_img, bottom[i]->channels(), sz, sz);
	}

	template <typename Dtype>
	static void rand_sample_patches(const Dtype* src, Dtype* dst, int nsample, int c, int h, int w, int sz)
	{
		for(int i=0; i<nsample; i++)
		{
			int x0 = rand() % (w-sz);
			int y0 = rand() % (h-sz);
			for(int ci=0; ci<c; ci++)
			{
				const Dtype* pSrc = src + ci*h*w + y0*w + x0;
				Dtype* pDst = dst + ci*sz*sz;
				for(int y=0; y<sz; y++)
				{
					for(int x=0; x<sz; x++)
						pDst[x] = pSrc[x];
					pSrc += w;
					pDst += sz;
				}
			}
		}
	}

	template <typename Dtype>
	Dtype PatchSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		const int n_per_img = layer_param_.patch_sample_param().sample_per_img();
		const int sz = layer_param_.patch_sample_param().patch_size();
		for(size_t i = 0; i<bottom.size(); i++)
		{
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = (*top)[i]->mutable_cpu_data();
			const int n = bottom[i]->num();
			const int c = bottom[i]->channels();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			const int cl = h*w;
			for(int ni=0; ni<n; ni++)
			{
				Dtype* pDst = top_data + ni*n_per_img*c*sz*sz;
				const Dtype* pSrc = bottom_data + ni*c*h*w;			
				rand_sample_patches(pSrc, pDst, n_per_img, c, h, w, sz);
			}
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(PatchSampleLayer);

}  // namespace caffe
