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
