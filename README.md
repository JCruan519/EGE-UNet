# EGE-UNet
This is the official code repository for "EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation", which is accpeted by *26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI2023)* as a regular paper!

**0. Main Environments**
- python 3.8
- [pytorch 1.8.0](https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-win_amd64.whl)
- [torchvision 0.9.0](https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl)

**1. Prepare the dataset.**

- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

**2. Train the EGE-UNet.**
```
cd EGE-UNet
```
```
python train.py
```

**3. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
