<h2 align="center">Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection</h2>
<p align="center">
    <!-- <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/YuLiu-61/ALDM/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/YuLiu-61/ALDM">
    </a>
    <a href="https://github.com/YuLiu-61/ALDM/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/YuLiu-61/ALDM">
    </a>
    <a href="https://github.com/YuLiu-61/ALDM/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/YuLiu-61/ALDM?color=pink">
    </a>
    <a href="https://github.com/YuLiu-61/ALDM">
        <img alt="issues" src="https://img.shields.io/github/stars/YuLiu-61/ALDM">
    </a>
</p>

---


<!-- This is the official implementation of the paper "[Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8856)" which has been accepted for publication in the IEEE Transactions on Automation Science and Engineering (T-ASE). -->
This is the official implementation of the paper "ALDM for Industrial TSAD". The paper has been accepted for publication in the IEEE Transactions on Automation Science and Engineering (T-ASE).


## 💡 Framework
![Framework](./asset/ALDM.png)

<!-- ## 🌋 Main results
<img src="./asset/results2.png" width="450" />
<img src="./asset/results1.png" width="900" /> -->

## 🏕️ Requirements
* matplotlib==3.7.5
* numpy==1.24.3
* pandas==2.0.3
* scikit_learn==0.24.1
* torch==2.1.0




```sh
pip install -r requirements.txt
```

## 📍 Data
We test our method for four public datasets, e.g., ```SWaT```, ```TEPE```, ```PSM```,and ```MSL```

[`SWaT`](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat) 
[`TEPE`](https://drive.google.com/drive/folders/1NlD8SxEQp7EMK9zhy8ibo6AIoz5ksh3o?usp=sharing) 
```sh
mkdir Dataset
cd Dataset
mkdir input
```
Download the dataset in ```Data/input```.
## 🔥 Train
- train for ALDM
```sh
For example, training for SWaT
```sh
python -u train.py --name SWaT --model ALDM --lr 0.002 --batch_size 512 --num_epochs 100 
```
- train for ```DeepSVDD```, ```DeepSAD```, ```DROCC```, and ```ALOCC```. 
```sh
python train_other_model.py --name SWaT --model DeepSVDD
```
- train for ```MTGFLOW```, ```USAD``` and ```DAGMM```
We report the results by the implementations in the following links: 

[`MTGFLOW`](https://github.com/zqhang/MTGFLOW),[`USAD`](https://github.com/manigalati/usad) and [`DAGMM`](https://github.com/danieltan07/dagmm/)


## 🚀 Test

For example, testing for SWaT 
```sh
sh runners/run_SWaT_test.sh
```

<!-- ## Acknowledgement
Some of the code in this repository is based on [`softclt`](https://github.com/seunghan96/softclt)and[`MTGFLOW`](https://github.com/zqhang/MTGFLOW). -->

## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```
@article{liu2026aldm,
  title={Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection},
  author={Liu, Yu and Song, Yifan and Shu, Shaolong and Lin, Feng and Wang, Jun and Guo, Yafeng},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026},
  publisher={IEEE}
}
```




