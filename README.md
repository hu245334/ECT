# ECT : Query Your Input for Fast Image deblurring

#### Congratulations on the release of our first deblur code!

    Congratulations! Our paper has been accepted for presentation at the IEEE World Congress on Computational Intelligence (IEEE WCCI 2024) to be held at  Pacifico Yokohama, Yokohama, Japan, 30 June - 5 July 2024.

The official pytorch implementation of the paper *[ECT: Efficient Cross Transformer for Image Deblurrin](https://edas.info/showPaper.php?m=1570991621 "Show paper")*

***

#### Chengwei Hu\*, Weiling Cai, Chen Zhao

> Image deblurring presents a complex challenge intending to renew visual clarity in images affected by camera shake or object motion. However, traditional deblurring methodologies tend to emphasize local features, ignoring critical contextual information, which consequently limits their efficacy in addressing common blurry image issues. This paper proposes Efficient Cross Transformer (ECT) to overcome the limitations of inadequate global features in image restoration across different scales. ECT designs the cross-attention layers to achieve interaction between input tokens and network tokens, aiming to efficiently capture image details in the input space and feature information in the latent space of each layer. The cross-attention mechanism works in latent layers, integrating image features from different scales while reducing the computational burden compared to traditional Transformers. Furthermore, ECT employs windowing techniques in a strategic manner to capture local hazy components, duly amplifying its image restoration abilities. Empirical evidence shows that ECT has achieved state-of-the-art results in deblurring images, demonstrating excellent performance without requiring a large training dataset. Consequently, ECT emerges as a promising solution for rectifying blurred images across artificial and real-world environments.







### News

**2023.08.15** The code are available now.

### Installation

This implementation based on [NAFNet](https://github.com/megvii-research/NAFNet/).

```python
python 3.9.5
pytorch 1.12.1
cuda 12.0
```































































### Results and Pre-trained Models

| Dataset    | PSNR    | SSIM   | pretrained models                                                                      |
| :--------- | :------ | :----- | :------------------------------------------------------------------------------------- |
| GoPro      | 32.8705 | 0.9606 | [Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1bCl7W0ccpjvYSqd54Pv4Uw?pwd=ebqj) |
| HIDE       | 33.7103 | 0.9668 | [Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1Y7uQQoJ2BJaZywkrXwjP3Q?pwd=v5qy) |
| RealBlur-J | 33.7103 | 0.9668 | [Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1vXgqFCdmIWNcI73aEeFL1Q?pwd=psf9) |
| RealBlur-R | 33.7103 | 0.9668 | [Google Drive]()  \|  [百度网盘](https://pan.baidu.com/s/1BLy2PBb_4jFFmcA7YAW_2A?pwd=jtpm) |

### Citations

If NAFNet helps your research or work, please consider citing ECT.

















### Contact

If you have any questions, please contact <EliHill@163.com>

***

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=hu245334/ECT)

</details>****

