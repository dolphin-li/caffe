#pragma once

#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

//#define IMAGE_DEBUG

#ifdef IMAGE_DEBUG
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

namespace caffe {

	template <typename Dtype>
	__global__ void rgb2luma_forward(Dtype* dst, const Dtype*src,
		const int nsample,
		const int npixel, const int nchannel, 
		const Dtype Mb, const Dtype Mg, const Dtype Mr, const Dtype T) 
	{
		CUDA_KERNEL_LOOP(index, npixel*nsample) {
			const int isample = index/npixel;
			const int ipixel = index%npixel;
			const int srcIndex = isample*npixel*nchannel + ipixel;
			dst[index] = Mb * src[srcIndex] + Mg * src[srcIndex+npixel] + Mr * src[srcIndex+npixel+npixel] + T;
		}
	}

	template <typename Dtype>
	Dtype Bgr2LumaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		const Dtype scale = layer_param_.bgr2luma_param().scale();
		// bgr
		const Dtype M[3] = {24.966f/255.f, 128.553f/255.f, 65.481f/255.f};
		const Dtype T = 16.f * scale;

		for(size_t i = 0; i<bottom.size(); i++)
		{
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = (*top)[i]->mutable_gpu_data();
			const int n = bottom[i]->num();
			const int c = bottom[i]->channels();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			const int cl = h*w;

			rgb2luma_forward<Dtype><<<CAFFE_GET_BLOCKS(cl*n), CAFFE_CUDA_NUM_THREADS>>>(
				top_data, bottom_data, n, cl, c, M[0], M[1], M[2], T);
			CUDA_POST_KERNEL_CHECK;

#ifdef IMAGE_DEBUG
			cv::Mat img;
			img.create(h, w, CV_8UC1);
			for(int ni=0; ni<n; ni++)
			{
				const Dtype* pSrc = (*top)[i]->cpu_data() + ni*h*w;
				for(int p=0; p<cl; p++)
				{
					img.at<char>(p/w, p%w) = pSrc[p]*255;
				}
				cv::imshow("luma", img);
				cvWaitKey();
			}
#endif

		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(Bgr2LumaLayer);

}  // namespace caffe