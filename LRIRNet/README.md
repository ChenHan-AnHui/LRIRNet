# LRIRNet

Official implementation for ASOC paper :
“LRIRNet: A lightweight and real-time infrared small target detection network based on multi-cue”
<br><br>

## News & Updates
* **June 1, 2026: Pulic The LRIRNet Code.**
<br><br>

## Requirements
- **Python 3.7**
- **pytorch 1.10.0 or higher**
- **pip install -r requirements.txt**
<br><br>

## Datasets
* **NUAA-SIRST** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1k** &nbsp; [[download]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)

**We used the NUAA-SIRST, NUDT-SIRST, IRSTD-1k for both training and val. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1itdAc4u7jPr2zxJ6arRorA?pwd=y9sg) (key:y9sg) and unzip the file to the folder `./LRIRNet/`.** 

* **Our datesets have the following structure:**
  ```
  ├──./datasets/
  │    ├── NUAA-SIRST
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── Misc_2_pixels0.png
  │    │    │    ├── Misc_2_pixels1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── trainval.txt
  │    │    │    ├── test.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── trainval.txt
  │    │    │    ├── test.txt
  │    ├── IRSTD-1k
  │    │    ├── IRSTD1k_Img
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── IRSTD1k_Label
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── trainval.txt
  │    │    │    ├── test.txt
  ```
<be>
<br><br>

## Experiments 

The training and val experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 2080Ti GPU of 11 GB Memory.

The trained model results are in `./result`
<br><br>

## Training
IRSTD-1k:
```
python train.py 
```
<br><br>

## Citation

Please cite our paper in your publications if our work helps your research. BibTeX reference is as follows.

```
@article{CHEN2026115507,
title = {LRIRNet: A lightweight and real-time infrared small target detection network based on multi-cue},
journal = {Applied Soft Computing},
volume = {201},
pages = {115507},
year = {2026},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2026.115507},
url = {https://www.sciencedirect.com/science/article/pii/S1568494626009555},
author = {Han Chen and Ziqiang Cao and Jinpeng Lu and Jun Tang and Fei Sun and Yangyang Guo}
}
```
<br><br>

## Contact
**Welcome to raise issues or email to [WB24101001@stu.ahu.edu.cn](WB24101001@stu.ahu.edu.cn) for any question regarding our LRIRNet.**
