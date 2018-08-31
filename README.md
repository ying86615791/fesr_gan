**Tensorflow** implementation for **Facial Expression Synthesis using GAN**
Reference: [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow)
- 参考[AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow), 实现人脸表情编辑
- data.py中ImgDataPair类实现pair image + label的读取
- 其他多个模块参考[AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow)
- 使用identity preserve loss需要下载[vgg-face的MatConvNet预训练模型](http://www.vlfeat.org/matconvnet/models/vgg-face.mat)