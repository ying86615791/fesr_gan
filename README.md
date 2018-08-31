**Tensorflow** implementation for **Facial Expression Synthesis using GAN**
- Reference: [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow)
- 参考[AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow), 实现人脸表情编辑
- data.py中ImgDataPair类实现pair image + label的读取
- 其他多个模块参考[AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow)
- 使用identity preserve loss需要下载[vgg-face的MatConvNet预训练模型](http://www.vlfeat.org/matconvnet/models/vgg-face.mat)

- training

	- for 128x128 images

		```console
		python train.py --img_size 128 --experiment_name ck_cv1_shortcut0_dz_gencfc --epoch 2000
		
- testing

	- test with different expressions
	
		```console
		python test.py --experiment_name ck_cv1_shortcut0_dz_gencfc --n_slide 1
		
	- test with different expressions and intensity control
	
		```console
		python test.py --experiment_name ck_cv1_shortcut0_dz_gencfc --test_int_min -1.0 --test_int_max 1.0 --n_slide 10