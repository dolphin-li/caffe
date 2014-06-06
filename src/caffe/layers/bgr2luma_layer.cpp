#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void Bgr2LumaLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		CHECK_GE(bottom.size(), 1);		  
		CHECK_EQ(bottom.size(), top->size());
		CHECK_EQ(bottom[0]->channels(), 3) << "only BGR-3 channels accepted";
		CHECK_EQ((*top)[0]->channels(), 1) << "only 1-channel output accepted";

		for(size_t i=0; i<bottom.size(); i++)
			(*top)[i]->Reshape(bottom[i]->num(), 1, bottom[i]->height(), bottom[i]->width());
	}

	template <typename Dtype>
	Dtype Bgr2LumaLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		const Dtype scale = layer_param_.bgr2luma_param().scale();
		// r: 65.481, g: 128.553, b: 24.969
		const Dtype M[3] = {24.966f/255.f, 128.553f/255.f, 65.481f/255.f};
		const Dtype T = 16.f * scale;

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
				Dtype* pDst = top_data + ni*h*w;
				const Dtype* pSrc = bottom_data + ni*c*h*w;
				for(int p=0; p<cl; p++)
					pDst[p] = M[0]*pSrc[p] + M[1]*pSrc[p+cl] + M[2]*pSrc[p+cl+cl] + T;
			}
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(Bgr2LumaLayer);

}  // namespace caffe
