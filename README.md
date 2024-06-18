# M2Mamba (Under Review)
---
M2Mamba: Multi-Scale in Multi-Scale Mamba for Blind Face Restoration in Telemedicine Application


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


