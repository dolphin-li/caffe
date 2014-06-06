#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void CalcCovLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		CHECK_GE(bottom.size(), 1);		  
		CHECK_EQ(bottom.size(), top->size());

		for(size_t i=0; i<bottom.size(); i++)
			(*top)[i]->Reshape(bottom[i]->num(), 1, bottom[i]->channels(), bottom[i]->channels());
		multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
		Ex_.Reshape(1, 1, bottom[0]->channels, 1);
	}

	template <typename Dtype>
	Dtype CalcCovLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		for(size_t i = 0; i<bottom.size(); i++)
		{
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = (*top)[i]->mutable_cpu_data();
			const int n = bottom[i]->num();
			const int c = bottom[i]->channels();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			const int cl = h*w;
			Dtype scale = Dtype(1.0);
			if(layer_param_.calc_cov_param().scale_by_num())
				scale = Dtype(1)/Dtype(h*w);
			for(int ni=0; ni<n; ni++)
			{
				Dtype* pDst = top_data + ni*c*c;
				const Dtype* pSrc = bottom_data + ni*c*h*w;
				Dtype* exData = Ex_.mutable_cpu_data();
				const Dtype* mpData = multiplier_.cpu_data();

				// E[X]
				caffe_cpu_gemv(CblasNoTrans, c, cl, scale, pSrc, mpData, Dtype(0), exData);

				// E[X]E[X]'
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, c, c, 1, Dtype(1), exData, exData, Dtype(0.0), pDst);

				// E[XX'] - E[X]E[X]'
				caffe_cpu_gemm(CblasNoTrans, CblasTrans, c, c, cl, scale, pSrc, pSrc, Dtype(-1.0), pDst);
			}
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(CalcCovLayer);

}  // namespace caffe
