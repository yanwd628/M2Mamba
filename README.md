# M2Mamba (BIBM)
---
M2Mamba: Multi-Scale in Multi-Scale Mamba for Blind Face Restoration in Telemedicine Application

[Paper](https://ieeexplore.ieee.org/document/10822469) | [Project](https://github.com/yanwd628/M2Mamba)

## Dependencies
+ Python 3.6
+ PyTorch >= 1.7.0
+ matplotlib
+ opencv
+ torchvision
+ numpy
+ mamba


## Datasets are provided in [here](https://github.com/wzhouxiff/RestoreFormer?tab=readme-ov-file#preparations-of-dataset-and-models)


## Train and Test (based on [Basicsr](https://github.com/XPixelGroup/BasicSR))

    python m2mamba/train.py -opt options/train/xxx.yml --auto_resume
    python m2mamba/test.py -opt options/test/xxx.yml



**ps: the path configs should be changed to your own path and the pre-trained weights will be available soon**

Our pretrained model is available [Google Drive](https://drive.google.com/file/d/1uIxbKW5oh6AeeXL8jKHNxR0saU4_U_Q6/view?usp=sharing)


```
@INPROCEEDINGS{10822469,
  author={Yan, Weidan and Shao, Wenze and Zhang, Dengyin},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={M2Mamba: Multi-Scale in Multi-Scale Mamba for Blind Face Restoration}, 
  year={2024},
  volume={},
  number={},
  pages={3877-3882},
  keywords={Hands;Biological system modeling;Telemedicine;Semantics;Noise reduction;Stochastic processes;Transformers;Image restoration;Convolutional neural networks;Faces;blind face restoration;state space model;multi-scale;telemedicine},
  doi={10.1109/BIBM62325.2024.10822469}}
```
