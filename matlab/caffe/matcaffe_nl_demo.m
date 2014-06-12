addpath('D:\Git\caffe\bin');
addpath('C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin');

% init with GPU used
use_gpu = 1;
model_def_file = '../../bin/net/t1_matlab.prototxt';
model_file = '../../bin/weights/t1/_iter_2740000';
matcaffe_init(use_gpu, model_def_file, model_file);

% load image
im = imread('D:\data\BSDS500\train\2092.jpg');

% crop
crop_size = 256;
h_off = ceil((size(im,1)-crop_size)/2);
w_off = ceil((size(im,2)-crop_size)/2);
im = im(h_off:h_off+crop_size-1, w_off:w_off+crop_size-1, :);

% rgb2bgr
im = im(:,:,[3 2 1]);

% im2double
im = single(im)/single(255);

% add noises
im = im + single(randn(size(im))*20/255);

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
input_data = zeros(crop_size, crop_size, 3, 1, 'single');
input_data(:,:,:,1) = im;
patch_noise_level_estimated = caffe('forward', input_data);
noise_level_es = median(patch_noise_level_estimated{1}(:))
toc;
caffe('reset');




