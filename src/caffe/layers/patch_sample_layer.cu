#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

	template <typename Dtype>
	__global__ void rand_sample_patches_gpu(const Dtype* src, Dtype* dst, const unsigned int* randXY, int nsample, int c, int h, int w, int sz)
	{
		const int sz2 = sz*sz;
		CUDA_KERNEL_LOOP(index, nsample*sz2) {
			int isample = index / sz2;
			int ipixel = index % sz2;
			int ix = ipixel % sz;
			int iy = ipixel / sz;
			unsigned int x0 = randXY[isample*2] % (w-sz);
			unsigned int y0 = randXY[isample*2+1] % (h-sz);
			for(int ci=0; ci<c; ci++)
			{
				const Dtype* pSrc = src + ci*h*w + (y0+iy)*w + x0 + ix;
				Dtype* pDst = dst + ci*sz*sz + iy*sz + ix;
				*pDst = *pSrc;
			}
		}
	}

	template <typename Dtype>
	Dtype PatchSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		const int n_per_img = layer_param_.patch_sample_param().sample_per_img();
		const int sz = layer_param_.patch_sample_param().patch_size();
		unsigned int* rand_xy_buffer = 0;
		CUDA_CHECK( cudaMalloc<unsigned int>(&rand_xy_buffer, n_per_img*2) );
		shared_ptr<unsigned int> rand_xy(rand_xy_buffer);
		for(size_t i = 0; i<bottom.size(); i++)
		{
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = (*top)[i]->mutable_gpu_data();
			const int n = bottom[i]->num();
			const int c = bottom[i]->channels();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			for(int ni=0; ni<n; ni++)
			{
				Dtype* pDst = top_data + ni*n_per_img*c*sz*sz;
				const Dtype* pSrc = bottom_data + ni*c*h*w;
				CURAND_CHECK(curandGenerate(Caffe::curand_generator(), rand_xy.get(), n_per_img*2));
				rand_sample_patches_gpu<Dtype><<<CAFFE_GET_BLOCKS(n_per_img*sz*sz), CAFFE_CUDA_NUM_THREADS>>>
					(pSrc, pDst, rand_xy.get(), n_per_img, c, h, w, sz);
				CUDA_POST_KERNEL_CHECK;
			}
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(PatchSampleLayer);

}  // namespace caffe
