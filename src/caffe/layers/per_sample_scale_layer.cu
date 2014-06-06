#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	Dtype PerSampleScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		for(size_t i = 0; i<bottom.size(); i++)
		{
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = (*top)[i]->mutable_gpu_data();
			const Dtype* scalar_data = blobs_[i]->gpu_data();
			const int n = bottom[i]->num();
			const int c = bottom[i]->channels();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			const int cl = h*w;
			for(int ni=0; ni<n; ni++)
			{
				Dtype* pDst = top_data + ni*c*h*w;
				const Dtype* pSrc = bottom_data + ni*c*h*w;
				const Dtype* mpData = scalar_data + ni*w*h;

				caffe_cpu_dgmm(CUBLAS_SIDE_LEFT, cl, c, pSrc, mpData, pDst); 
			}
		}
		return Dtype(0.);
	}

	template <typename Dtype>
	void PerSampleScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{

	}

	INSTANTIATE_CLASS(PerSampleScaleLayer);

}  // namespace caffe
