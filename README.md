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
sh test_EDSR_x4.sh
```
Our pretrained model can be accessed from [Google Drive](https://drive.google.com/drive/folders/19sWPy0IHISnHX8T4g1zH8ZHVgISU89t_?usp=sharing).

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
