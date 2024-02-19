# DAQ: Channel-Wise Distribution-Aware Quantization for Deep Image Super-Resolution Networks

Official implementation of our WACV 2022 [paper](https://openaccess.thecvf.com/content/WACV2022/papers/Hong_DAQ_Channel-Wise_Distribution-Aware_Quantization_for_Deep_Image_Super-Resolution_Networks_WACV_2022_paper.pdf).

### Conda Environment setting
```
conda env create -f environment.yml --name DAQ
conda activate DAQ
conda install -c anaconda scikit-image
```

### Dependencies
* Python 3.6
* PyTorch == 1.1.0
* coloredlogs >= 14.0
* scikit-image


### Codes
Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).


### Train
```
sh train_EDSR_x4.sh
```
Pretrained model to start training from can be accessed from [Google Drive](https://drive.google.com/drive/folders/19sWPy0IHISnHX8T4g1zH8ZHVgISU89t_?usp=sharing).


### Test
```
sh test.sh edsr_baseline 2 2 4 (edsr_baseline w2a2qq4)
sh test.sh edsr_baseline 3 3 4 (edsr_baseline w3a3qq4)
sh test.sh edsr_baseline 4 4 4 (edsr_baseline w4a4qq4)
sh test.sh edsr_full 2 2 8 (edsr_full w2a2qq8)
```
Our pretrained model can be accessed from [Google Drive](https://drive.google.com/drive/folders/19sWPy0IHISnHX8T4g1zH8ZHVgISU89t_?usp=sharing).

## Additional Results

Our model achieves the following performance (PSNR / SSIM) when trained for 60 epochs :

  
| Model           | Precision (w/a) |     Set5      |     Set14     |     B100      |   Urban100    |
| --------------- |:---------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **EDSR-baseline** (x4)   |     32 / 32     | 32.10 / 0.894 | 28.58 / 0.781 | 27.56 / 0.736 | 26.04 / 0.785 |
| EDSR-baseline-DAQ        |      4 / 4      | 31.85 / 0.887 | 28.38 / 0.776 | 27.42 / 0.732 | 25.73 / 0.772 |
| EDSR-baseline-DAQ        |      3 / 3      | 31.66 / 0.884 | 28.19 / 0.771 | 27.28 / 0.728 | 25.40 / 0.762 |
| EDSR-baseline-DAQ        |      2 / 2      | 31.01 / 0.871 | 27.89 / 0.762 | 27.09 / 0.719 | 24.88 / 0.740 |


| Model           | Precision (w/a) |     Set5      |     Set14     |     B100      |   Urban100    |
| --------------- |:---------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **RDN** (x4)    |     32 / 32     | 32.24 / 0.896 | 28.67 / 0.784 | 27.63 / 0.738 | 26.29 / 0.792 |
| RDN-DAQ         |      4 / 4      | 31.91 / 0.889 | 28.38 / 0.775 | 27.38 / 0.733 | 25.81 / 0.779 | 
| RDN-DAQ         |      3 / 3      | 31.57 / 0.883 | 28.18 / 0.771 | 27.27 / 0.728 | 25.47 / 0.765 | 
| RDN-DAQ         |      2 / 2      | 30.71 / 0.866 | 27.61 / 0.755 | 26.93 / 0.715 | 24.71 / 0.731 | 


| Model           | Precision (w/a) |     Set5      |     Set14     |     B100      |   Urban100    |
| --------------- |:---------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **SRResNet** (x4)|     32 / 32     | 32.07 / 0.893 | 28.50 / 0.780 | 27.52 / 0.735 | 25.86 / 0.779 |
| SRResNet-DAQ    |      4 / 4      | 31.85 / 0.889 | 28.41 / 0.777 | 27.45 / 0.732 | 25.70 / 0.772 | 
| SRResNet-DAQ    |      3 / 3      | 31.81 / 0.889 | 28.35 / 0.776 | 27.40 / 0.733 | 25.63 / 0.772 | 
| SRResNet-DAQ    |      2 / 2      | 31.57 / 0.886 | 28.19 / 0.773 | 27.30 / 0.729 | 25.39 / 0.765 | 


### Citation
```
@inproceedings{hong2022daq,
  title={DAQ: Channel-Wise Distribution-Aware Quantization for Deep Image Super-Resolution Networks},
  author={Hong, Cheeun and Kim, Heewon and Baik, Sungyong and Oh, Junghun and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2675--2684},
  year={2022}
}
```

### Contact
Email : cheeun914@snu.ac.kr 
