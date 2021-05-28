# DoWnGAN
This repo is under development as thesis work and is by not complete or tested.

This repo implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to perform single image super resolution (SISR) to downscale climate fields.

# Requirements
Although not absolutely required to run this source code, a working instance of CUDA and Pytorch is required for this repo. 

# Check if CUDA is installed and PyTorch has access to cuda GPU
```python3
torch.cuda.is_available()
```

If the above returns True, then PyTorch is accessing the GPU.

 # References
 Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. 2017. “Improved Training of Wasserstein GANs.” ArXiv:1704.00028 [Cs, Stat], December. http://arxiv.org/abs/1704.00028.
Arjovsky, Martin, Soumith Chintala, and Léon Bottou. 2017. “Wasserstein GAN.” ArXiv:1701.07875 [Cs, Stat], December. http://arxiv.org/abs/1701.07875.
Ledig, Christian, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, et al. 2017. “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.” ArXiv:1609.04802 [Cs, Stat], May. http://arxiv.org/abs/1609.04802.
Wang, Xintao, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, and Xiaoou Tang. 2018. “ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.” ArXiv:1809.00219 [Cs], September. http://arxiv.org/abs/1809.00219.
