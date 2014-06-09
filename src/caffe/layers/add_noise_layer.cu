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
	Dtype AddNoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) 
	{
		for(size_t i = 0; i<bottom.size(); i++)
		{
			// generate noise-levels
			generate_noise_levels();
			const Dtype* noise_level_data = noise_levels_.gpu_data();
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = (*top)[i*2]->mutable_gpu_data();
			Dtype* top_nl = (*top)[i*2+1]->mutable_gpu_data();
			Dtype* noise_data = noises_.mutable_gpu_data();

			const int n = bottom[i]->num();
			const int np = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
			caffe_gpu_rng_gaussian(n*np, Dtype(0), Dtype(1), noise_data);
			caffe_gpu_dgmm(CUBLAS_SIDE_RIGHT, np, n, noise_data, noise_level_data, noise_data);
			caffe_gpu_add(n*np, bottom_data, noise_data, top_data);

			caffe_gpu_copy(noise_levels_.count(), noise_levels_.gpu_data(), top_nl);
#ifdef IMAGE_DEBUG
			const Dtype scale = layer_param_.add_noise_param().scale();
			const int h = bottom[i]->height();
			const int w = bottom[i]->width();
			const int c = bottom[i]->channels();
			cv::Mat img;
			img.create(h, w*2, CV_MAKETYPE(CV_8U, c));
			for(int in=0; in<n; in++)
			{
				const Dtype* clean = bottom[i]->cpu_data() + in*np;
				const Dtype* noised = (*top)[i]->cpu_data() + in*np;
				Dtype sum = 0.f, sum2 = 0.f;
				for(int j=0; j<np; j++)
				{
					Dtype ns = noised[j]-clean[j];
					sum += ns;
					sum2 += ns*ns;

					int ci = j/(h*w);
					int y = j%(h*w)/w;
					int x = j%(h*w)%w;
					img.at<unsigned char>(y,x) = std::min(Dtype(255), std::max(Dtype(0), clean[j]/scale));
					img.at<unsigned char>(y,x+w) = std::min(Dtype(255), std::max(Dtype(0), noised[j]/scale));
				}
				Dtype cov = sum2/np - (sum/np)*(sum/np);
				Dtype std = sqrt(cov);
				LOG(INFO) << "nl: " << noise_levels_.cpu_data()[in]/scale << ", " << std/scale;
				cv::imshow("noised", img);
				cvWaitKey();
			}
#endif
		}
		return Dtype(0.);
	}

	INSTANTIATE_CLASS(AddNoiseLayer);

}  // namespace caffe
