# ECT : Query Your Input for Fast Image deblurring
#### Congratulations on the release of our first deblur code!
The official pytorch implementation of the paper **[Query Your Input for Fast Image deblurring ()]()**

---

#### Chengwei Hu\*, Weiling Cai, Chen Zhao, Ziqi Zeng

>Image deblurring is a challenging task, involving the restor-tion of sharpness in single blurred images caused by camera shake or object movement. Conventional deblurring methods often focus on local features to reduce computational complexity. However, most of these approach lack the con-sideration of crucial background information, leading to an inability to accurately address real-world ambiguity problems. To overcome this limitation, we propose an innovative transformer-based image deblurring model called Efficient Cross Transformer (ECT). In ECT, the input tokens utilize the cross-attention layer to attend to the tokens in the network, and then pay attention to each other via the self-attention layer, effectively capturing global information. This effective integration of global features greatly reduces computational overhead. Furthermore, ECT leverages a windowing technique to consider the local characteristics and size of the blur, resulting in reduced computational burden and improved image restoration capabilities. Experimental results demonstrate that ECT outperforms traditional transformer methods, achieving state-of-the-art deblurring performance without the need for an extensive training dataset. ECT represents an efficient solution for image deblurring, making it a promising tool for processing blurred images in both synthetic and real-world scenes.

<!--| <img src="./figures/denoise.gif"  height=224 width=224 alt="NAFNet For Image Denoise"> | <img src="./figures/deblur.gif" width=400 height=224 alt="NAFNet For Image Deblur"> | <img src="./figures/StereoSR.gif" height=224 width=326 alt="NAFSSR For Stereo Image Super Resolution"> |-->
<!--| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |-->
<!--|                           GoPro                            |                            HIDE                            |                           RealWorld                           |-->


### News
**2023.08.15** The code are available now.


### Installation
This implementation based on [NAFNet](https://github.com/megvii-research/NAFNet/).

```python
python 3.9.5
pytorch 1.12.1
cuda 12.0
```


<!--### Quick Start -->
<!--* Image Denoise Colab Demo: [<a href="https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing)-->
<!--* Image Deblur Colab Demo: [<a href="https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing)-->
<!--* Stereo Image Super-Resolution Colab Demo: [<a href="https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing)-->
<!--* Single Image Inference Demo:-->
<!--    * Image Denoise:-->
<!--    ```-->
<!--    python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png-->
<!--  ```-->
<!--    * Image Deblur:-->
<!--    ```-->
<!--    python basicsr/demo.py -opt options/test/REDS/NAFNet-width64.yml --input_path ./demo/blurry.jpg --output_path ./demo/deblur_img.png-->
<!--    ```-->
<!--    * ```--input_path```: the path of the degraded image-->
<!--    * ```--output_path```: the path to save the predicted image-->
<!--    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded. -->
<!--    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for single image restoration[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFNet)-->
<!--* Stereo Image Inference Demo:-->
<!--    * Stereo Image Super-resolution:-->
<!--    ```-->
<!--    python basicsr/demo_ssr.py -opt options/test/NAFSSR/NAFSSR-L_4x.yml \-->
<!--    --input_l_path ./demo/lr_img_l.png --input_r_path ./demo/lr_img_r.png \-->
<!--    --output_l_path ./demo/sr_img_l.png --output_r_path ./demo/sr_img_r.png-->
<!--    ```-->
<!--    * ```--input_l_path```: the path of the degraded left image-->
<!--    * ```--input_r_path```: the path of the degraded right image-->
<!--    * ```--output_l_path```: the path to save the predicted left image-->
<!--    * ```--output_r_path```: the path to save the predicted right image-->
<!--    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded. -->
<!--    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for stereo image super-resolution[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFSSR)-->
<!--* Try the web demo with all three tasks here: [![Replicate](https://replicate.com/megvii-research/nafnet/badge)](https://replicate.com/megvii-research/nafnet)-->

### Results and Pre-trained Models

| Dataset|PSNR|SSIM| pretrained models |
|:----|:----|:----|:----|
|GoPro|32.8705|0.9606|[Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1bCl7W0ccpjvYSqd54Pv4Uw?pwd=ebqj)|
|HIDE|33.7103|0.9668|[Google Drive]()  \|  [百度网盘](链接：https://pan.baidu.com/s/1Y7uQQoJ2BJaZywkrXwjP3Q?pwd=v5qy)|
|RealBlur-J|33.7103|0.9668|[Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1vXgqFCdmIWNcI73aEeFL1Q?pwd=psf9 )|
|RealBlur-R|33.7103|0.9668|[Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1BLy2PBb_4jFFmcA7YAW_2A?pwd=jtpm )|



### Citations
If NAFNet helps your research or work, please consider citing ECT.

<!--```-->
<!--@article{chen2022simple,-->
<!--  title={Simple Baselines for Image Restoration},-->
<!--  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},-->
<!--  journal={arXiv preprint arXiv:2204.04676},-->
<!--  year={2022}-->
<!--}-->
<!--```-->


### Contact

If you have any questions, please contact EliHill@163.com

---

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=hu245334/ECT)

</details>****
