# Attention-Guided Pyramid Context Networks

[![](https://img.shields.io/badge/building-Done-green.svg?style=flat-square)](https://github.com/Tianfang-Zhang/AGPCNet)
![](https://img.shields.io/badge/language-PyTorch-blue.svg?style=flat-square)
[![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](./LICENSE)

> [Paper Link](https://ieeexplore.ieee.org/document/10024907)   
> Authors: Tianfang Zhang, Lei Li, Siying Cao, Tian Pu and Zhenming Peng  

## [Datasets](#attention-guided-pyramid-context-networks)

- MDFA dataset is available at [MDvsFa cGan](https://github.com/wanghuanphd/MDvsFA_cGAN).
- SIRST dataset is available at [SIRST](https://github.com/YimianDai/sirst).
- The SIRST Augment dataset: download from [Google Drive](https://drive.google.com/file/d/13hhEwYHU19oxanXYf-wUpZ7JtiwY8LuT/view?usp=sharing) or [BaiduYun Drive](https://pan.baidu.com/s/1c35pADjPhkAcLwmU-u0RBA) with code `ojd4`.

## [Usage](#attention-guided-pyramid-context-networks)

### Train
```python
python train.py --net-name agpcnet_1 --batch-size 8 --save-iter-step 20 --dataset mdfa
```

```python
python train.py --net-name agpcnet_1 --batch-size 8 --save-iter-step 40 --dataset sirstaug
```

```python
python train.py --net-name agpcnet_1 --batch-size 8 --save-iter-step 100 --dataset merged
```

### Inference

```python
python inference.py --pkl-path {checkpoint path} --image-path {image path}
```

### Evaluation
```python
python evaluation.py --dataset {dataset name} 
                     --sirstaug-dir {base dir of sirstaug}
                     --mdfa-dir {base dir of MDFA}
                     --pkl-path {checkpoint path}
```


## [Results](#attention-guided-pyramid-context-networks)

| Methods | Data      | Precision | Recall | mIoU   | Fmeasure | AUC    | Download | 
| :---:   | :---:     | :---:     | :---:  | :---:  | :---:    | :---:  | :---:    |
| AGPCNet+ResNet18 | MDFA      | 0.5939    | 0.7241 | 0.4843 | 0.6525   | 0.8682 | [model](./results/mdfa_mIoU-0.4843_fmeasure-0.6525.pkl) |
| AGPCNet+ResNet18 | SIRST Aug | 0.8323    | 0.8542 | 0.7288 | 0.8431   | 0.9344 | [model](./results/sirstaug_mIoU-0.7288_fmeasure-0.8431.pkl) |
| AGPCNet+ResNet18 | Merged    | 0.7453    | 0.8384 | 0.6517 | 0.7891   | 0.9194 | [model](./results/merged_mIoU-0.6517_fmeasure-0.7891.pkl) |



<center class="half">
    <img src="./roc.jpg" width=100%/>
</center>
<center style="font-size:16px;color:#C0C0C0;text-decoration:underline">Figure. Illustration of ROC compared with state-of-the-art methods.</center>


Evaluation of model-driven algorithms based on traditional metrics refers [ISTD-python].

## [Citation](#attention-guided-pyramid-context-networks)

```python
@article{zhang2023attention,
  title={Attention-guided pyramid context networks for detecting infrared small target under complex background},
  author={Zhang, Tianfang and Li, Lei and Cao, Siying and Pu, Tian and Peng, Zhenming},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2023},
  doi={10.1109/TAES.2023.3238703},
  publisher={IEEE}
}
```