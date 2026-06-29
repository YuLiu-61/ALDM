<h2 align="center">Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection</h2>
<p align="center">
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
        <img alt="stars" src="https://img.shields.io/github/stars/YuLiu-61/ALDM">
    </a>
</p>

---

Official implementation of ALDM for industrial time-series anomaly detection (TSAD).  
The paper has been accepted by IEEE Transactions on Automation Science and Engineering (T-ASE).

## 💡 Overview
- ALDM models latent distributions adaptively to capture complex normal patterns in multivariate industrial signals.
- Supports training and evaluation on four public datasets: SWaT, TEPE, PSM, and MSL.

![Framework](./asset/ALDM.png)

## 🏕️ Requirements
- torch==2.1.0
- numpy==1.24.3
- pandas==2.0.3
- scikit_learn==0.24.1
- matplotlib==3.7.5

Install all dependencies:

```sh
pip install -r requirements.txt
```

## 📍 Data Setup
ALDM expects raw/label files under `Dataset/Data/input` with subfolders per dataset:

```sh
mkdir -p Dataset/Data/input/{SWaT,TEPE,PSM,MSL}
```

Download datasets and place files accordingly:
- SWaT: [`link`](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat) 
- TEPE: [`link`](https://drive.google.com/drive/folders/1NlD8SxEQp7EMK9zhy8ibo6AIoz5ksh3o?usp=sharing)

Notes:
- Large raw files are not tracked in Git by default. Keep datasets locally at `Dataset/Data/input/...`.
- Processed files are produced under `Dataset/Data/input/processed` during preprocessing/training.

## 🔥 Training
Train ALDM on SWaT:

```sh
python -u main.py --name SWaT --model ALDM --lr 0.002 --batch_size 512 --epochs 100
```

Train baselines (```DeepSVDD```, ```DeepSAD```):

```sh
python train_other_model.py --name SWaT --model DeepSVDD
```

For MTGFLOW / USAD / DAGMM, we reference the official implementations:
- MTGFLOW: https://github.com/zqhang/MTGFLOW  
- USAD: https://github.com/manigalati/usad  
- DAGMM: https://github.com/danieltan07/dagmm

## Testing
Example (SWaT):

```sh
sh runners/run_SWaT_test.sh
```

## Citation
If you find this work useful, please cite:

```
@ARTICLE{liu2026aldm,
  author={Liu, Yu and Song, Yifan and Shu, Shaolong and Lin, Feng and Wang, Jun and Guo, Yafeng},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={Adaptive Latent Distribution Modeling for Industrial Time Series Anomaly Detection}, 
  year={2026},
  volume={23},
  number={},
  pages={7893-7907},
  doi={10.1109/TASE.2026.3674236}}
```
