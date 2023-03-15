# Speed Improvement of PAN++

## Ver 1. [(Conventional Version, Vanilla PAN++)](https://github.com/Zerohertz/pan_pp.pytorch/commit/02b70de62d8c9d58cd240e3b95cda82a16edf0b5)

> [./models/head/pan_pp_det_head.py/PAN_PP_DetHead.get_results()](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/models/head/pan_pp_det_head.py)

||g0|g1|g2|g3|g4|Total|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Time [ms]|34.647|355.830|1.874|985.164|0.007|1377.522|
|Weight [%]|2.515|25.831|0.136|71.517|0.001|-|

> [./models/post_processing/pa/pa.pyx/_pa()](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/models/post_processing/pa/pa.pyx)

||s0|s1|s2|s3|Total|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Time [ms]|2.800|336.512|7.402|2.780|349.495|
|Weight [%]|0.801|96.285|2.118|0.796|-|

## Ver 2. [(Update: ./models/post_processing/pa/pa.pyx)](https://github.com/Zerohertz/pan_pp.pytorch/commit/795a066f2730a68b02443e4d3bdbb004b1d98046)

<details>
<summary>Read more</summary>
<div>

> [./models/head/pan_pp_det_head.py/PAN_PP_DetHead.get_results()](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/models/head/pan_pp_det_head.py)

||g0|g1|g2|g3|g4|Total|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Time [ms]|35.272|270.528|1.678|994.205|0.008|1301.691|
|Weight [%]|2.710|20.783|0.129|76.378|0.001|-|

> [./models/post_processing/pa/pa.pyx/_pa()](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/models/post_processing/pa/pa.pyx)

||s0|s1|s2|s3|Total|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Time [ms]|2.420|256.450|7.560|2.807|269.277|
|Weight [%]|0.900|95.236|2.807|1.057|-|

<img src="https://user-images.githubusercontent.com/42334717/220261919-02c721a6-bfd6-47f6-b4cc-ff514dd65356.png">

</div>
</details>

## Ver 3. (Add: `resize_const`, `pos_const`, `len_const`)

<details>
<summary>Read more</summary>
<div>

- <a href='https://github.com/Zerohertz/pan_pp.pytorch/commit/e9ea507c091fdf1289df16184e4d30d911e9af4d'>resize_const, pos_const, len_const</a>
- <a href='https://github.com/Zerohertz/pan_pp.pytorch/commit/67c6aade111dd45e35be720f2b2c12319a83824c'>Full Factorial Design</a>

### [Full factorial design: Experiment 1](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/results/Ver3_1_2-4_0-0.2_0-1/test_ff.sh)

<img src="https://user-images.githubusercontent.com/42334717/220262002-90843c0f-69b3-4922-b34f-9ada34377350.png">
<img src="https://user-images.githubusercontent.com/42334717/220262009-9c7d256a-26b9-4c66-9715-3953bc06968d.png">
<img src="https://user-images.githubusercontent.com/42334717/220262016-6e2ae02e-7f7c-4dcc-ba22-6d2ceeaa2e34.png">
<img src="https://user-images.githubusercontent.com/42334717/220262161-90220aba-3ac3-4270-bd15-06cab1350c9b.png">
<img src="https://user-images.githubusercontent.com/42334717/220262150-1fb55a8d-1142-4579-aa29-8f6aed07a9ca.png">
<img src="https://user-images.githubusercontent.com/42334717/220262253-abd8314d-232e-4ecc-aa68-3150127a8e6a.png">

### [Full factorial design: Experiment 2](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/results/Ver3_2_2_0.2-0.3_0.4-0.6/test_ff.sh)

<img src="https://user-images.githubusercontent.com/42334717/220262413-0cd877a9-26f3-484c-b299-78c3941fea81.png">
<img src="https://user-images.githubusercontent.com/42334717/220262411-1b512ecd-da52-487e-be7b-91161a3b26ae.png">
<img src="https://user-images.githubusercontent.com/42334717/220262417-0927deb2-2ae2-43de-b819-8f09a077ae36.png">

</div>
</details>

## Ver 4. [(Update: ./models/post_processing/pa/pa.pyx)](https://github.com/Zerohertz/pan_pp.pytorch/commit/7127027bb90084bf49856aef0295409b6744518e)

<details>
<summary>Read more</summary>
<div>

<img src="https://user-images.githubusercontent.com/42334717/220262538-6dd26c0b-9d26-4529-af11-5fc1b430501c.png">

</div>
</details>

## Ver 5. [(Add: ./models/post_processing/boxgen/boxgen.pyx)](https://github.com/Zerohertz/pan_pp.pytorch/commit/233bf1dadb73673d255abb4ce5f3a86cd63636b6)

<details>
<summary>Read more</summary>
<div>

<img src="https://user-images.githubusercontent.com/42334717/220262542-14aead80-03ef-48f2-b0f9-3d27862a48a0.png">

</div>
</details>

## Ver 6. (Update: ./models/post_processing/)

<details>
<summary>Read more</summary>
<div>

- <a href='https://github.com/Zerohertz/pan_pp.pytorch/commit/dd64dc0c6db19044da24cd558f49849628278b3d'>Pixel Aggregation</a>
- <a href='https://github.com/Zerohertz/pan_pp.pytorch/commit/2fcae8ab733a63be1a38e47a577819a74d821d0c'>Boxgen</a>

### [Full factorial design: Experiment 1](https://github.com/Zerohertz/pan_pp.pytorch/blob/master/results/Ver6_1_2_0.2-0.3_0.4-0.6/test_ff.sh)

<img src="https://user-images.githubusercontent.com/42334717/220262771-55f3ac6e-e5ec-44b8-a63a-4367c081c1fd.png">
<img src="https://user-images.githubusercontent.com/42334717/220262769-426ce52e-6742-4224-8098-60e6a917cf71.png">
<img src="https://user-images.githubusercontent.com/42334717/220262766-2255b3c1-0a5f-4d12-a722-22cf2d2e71d4.png">
<img src="https://user-images.githubusercontent.com/42334717/220262761-804bf8f2-490d-47f7-ab16-e3a40db7700e.png">

</div>
</details>

## Ver 7. [(Update: ./models/post_processing/boxgen/boxgen.pyx)](https://github.com/Zerohertz/pan_pp.pytorch/commit/ca252f6b581878fa735a414c168e09522b97c171)

<details>
<summary>Read more</summary>
<div>

<img src="https://user-images.githubusercontent.com/42334717/220262937-b9e3a5ab-e13d-4c16-9dcf-7cce7fa606fb.png">

</div>
</details>

## Ver 8. [(Update: ./models/post_processing/)](https://github.com/Zerohertz/pan_pp.pytorch/commit/f3601b043948de1b3496295d87ea92e621f827bb)

<details open>
    <summary>Result</summary>
    <div>
        <img src="https://user-images.githubusercontent.com/42334717/220262939-4217d7bd-c5a2-49fe-a94d-c8022b6982a9.png">
    </div>
</details>

# Conclusion

![Conclusion 1](https://user-images.githubusercontent.com/42334717/220262931-1876d7e0-7453-4a6a-be61-0e2bb3963fac.png)

---

<details>
<summary>Read more</summary>
<div>

## News
- (2022/12/08) We will release the code and models of FAST in [link](https://github.com/czczup/FAST).
- (2022/10/09) We release stabler code for PAN++, see [pan_pp_stable](https://github.com/whai362/pan_pp_stable).
- (2022/04/22) Update PAN++ ICDAR 2015 joint training & post-processing with vocabulary & visualization code.
- (2021/11/03) Paddle implementation of PAN, see [Paddle-PANet](https://github.com/simplify23/Paddle-PANet). Thanks @simplify23.
- (2021/04/08) PSENet and PAN are included in [MMOCR](https://github.com/open-mmlab/mmocr).

## Introduction
This repository contains the official implementations of [PSENet](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html), [PAN](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Efficient_and_Accurate_Arbitrary-Shaped_Text_Detection_With_Pixel_Aggregation_Network_ICCV_2019_paper.html), [PAN++](https://arxiv.org/abs/2105.00405).

<details open>
<summary>Text Detection</summary>

- [x] [PSENet (CVPR'2019)](config/psenet/)
- [x] [PAN (ICCV'2019)](config/pan/)
- [x] [FAST (Arxiv'2021)](config/fast/)
</details>

<details open>
<summary>Text Spotting</summary>

- [x] [PAN++ (TPAMI'2021)](config/pan_pp)

</details>

## Installation

First, clone the repository locally:

```shell
git clone https://github.com/whai362/pan_pp.pytorch.git
```

Then, install PyTorch 1.1.0+, torchvision 0.3.0+, and other requirements:

```shell
conda install pytorch torchvision -c pytorch
pip install -r requirement.txt
```

Finally, compile codes of post-processing:

```shell
# build pse and pa algorithms
sh ./compile.sh
```

## Dataset
Please refer to [dataset/README.md](dataset/README.md) for dataset preparation.

## Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```

## Testing

### Evaluate the performance

```shell
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
cd eval/
./eval_{DATASET}.sh
```
For example:
```shell
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar
cd eval/
./eval_ic15.sh
```

### Evaluate the speed

```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar --report_speed
```

### Visualization

```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --vis
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar --vis
```


## Citation

Please cite the related works in your publications if it helps your research:

### PSENet

```
@inproceedings{wang2019shape,
  title={Shape Robust Text Detection with Progressive Scale Expansion Network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

### PAN

```
@inproceedings{wang2019efficient,
  title={Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network},
  author={Wang, Wenhai and Xie, Enze and Song, Xiaoge and Zang, Yuhang and Wang, Wenjia and Lu, Tong and Yu, Gang and Shen, Chunhua},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8440--8449},
  year={2019}
}
```

### PAN++

```
@article{wang2021pan++,
  title={PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Liu, Xuebo and Liang, Ding and Zhibo, Yang and Lu, Tong and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

### FAST

```
@misc{chen2021fast,
  title={FAST: Searching for a Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation}, 
  author={Zhe Chen and Wenhai Wang and Enze Xie and ZhiBo Yang and Tong Lu and Ping Luo},
  year={2021},
  eprint={2111.02394},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).

</div>
</details>
