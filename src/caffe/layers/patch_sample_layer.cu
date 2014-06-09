#pragma once

#include <cuda_runtime.h>

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

//#define IMAGE_DEBUG
#ifdef IMAGE_DEBUG
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

namespace caffe {

	template <typename Dtype>
	__global__ void rand_sample_patches_gpu(const Dtype* src, Dtype* dst, const unsigned int* randXY, int nimg, int nsample, int c, int h, int w, int sz)
	{
		const int sz2 = sz*sz;
		const int pixel_per_img_sample = nsample*sz2;
		CUDA_KERNEL_LOOP(index, nimg*pixel_per_img_sample) {
			const int iimg = index / pixel_per_img_sample;
			const int idx_per_img = index - iimg*pixel_per_img_sample;
			const int isample = idx_per_img / sz2;
			const int ipixel = idx_per_img - sz2 * isample;
			const int iy = ipixel / sz;
			const int ix = ipixel - sz * iy;
			const int rndpos = iimg * nsample + isample;
			unsigned int x0 = randXY[rndpos*2] % (w-sz);
			unsigned int y0 = randXY[rndpos*2+1] % (h-sz);

			const Dtype* pSrc = src + iimg*c*h*w + (y0+iy)*w + x0 + ix;
			Dtype* pDst = dst + iimg*pixel_per_img_sample  + isample*sz2*c + iy*sz + ix;
			for(int ci=0; ci<c; ci++)
				pDst[ci*sz2] = pSrc[ci*h*w];
		}
	}

	template <typename Dtype>
	Dtype PatchSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		const int n_per_img = layer_param_.patch_sample_param().sample_per_img();
		const int sz = layer_param_.patch_sample_param().patch_size();
		for(size_t i = 0; i<bottom.size(); i++)
		{
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = (*top)[i]->mutable_gpu_data();
			const int n = bottom[i]->num();
			const int c = bottom[i]->channels();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			unsigned int* rand_xy_buffer = 0;
			CUDA_CHECK( cudaMalloc<unsigned int>(&rand_xy_buffer, n*n_per_img*2*sizeof(unsigned int)) );

			CURAND_CHECK(curandGenerate(Caffe::curand_generator(), rand_xy_buffer, n*n_per_img*2));
			rand_sample_patches_gpu<Dtype><<<CAFFE_GET_BLOCKS(n*n_per_img*sz*sz), CAFFE_CUDA_NUM_THREADS>>>
				(bottom_data, top_data, rand_xy_buffer, n, n_per_img, c, h, w, sz);
			CUDA_POST_KERNEL_CHECK;

#ifdef IMAGE_DEBUG
			const int cl = sz*sz;
			cv::Mat img;
			img.create(h, w, CV_8UC1);
			std::vector<unsigned int> rand_xy_cpu(n*n_per_img*2);
			CUDA_CHECK( cudaMemcpy(rand_xy_cpu.data(), rand_xy_buffer, n*n_per_img*2*sizeof(unsigned int), cudaMemcpyDeviceToHost) );
			for(int ni=0; ni<(*top)[i]->num(); ni++)
			{
				const Dtype* pSrc = (*top)[i]->cpu_data() + ni*sz*sz*c;
				for(int p=0; p<cl; p++)
				{
					unsigned int x0 = rand_xy_cpu[ni*2] % (w-sz);
					unsigned int y0 = rand_xy_cpu[ni*2+1] % (h-sz);
					img.at<char>(p/sz + y0, p%sz + x0) = pSrc[p]*255;
				}
				if(ni%n_per_img==n_per_img-1)
				{
					cv::imshow("patch", img);
					memset(img.data, 0, img.dataend-img.data);
					cvWaitKey();
				}
			}
#endif

			CUDA_CHECK( cudaFree(rand_xy_buffer) );
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(PatchSampleLayer);

}  // namespace caffe
