## Global, Multi-Scale Standing Deadwood Segmentation In Centimeter-Scale Aerial Images

This repository contains code for the paper "Global, Multi-Scale Standing Deadwood Segmentation In Centimeter-Scale Aerial Images". https://doi.org/10.1016/j.ophoto.2025.100104


- Preprocessig: see `preprocessing_notebooks`
- Training:`launch_training_phaestos.sh`: This launches the `train.py` with the accelerate library and with the config `config_mixvision_large_oversample_newdata_full_training.json`
- Figures: see `plotting_notebooks`


This repository does not contain code for the inference. For that checkout: https://github.com/cmosig/deadtreesmodels


## Adaptations
* slight modifications of the train_dataset.py file
* deadtrees_preprocessing_dataset.py > added file to convert UAV orthophoto's and polygon labels to image tiles and masks + metadata.csv file, which can be used as a dataset for training/testing.
* train_adapted.py > copy and adapted file of train.py, to train smaller deep learning segmentation models on a 1 GPU windows computer.
* transfer_learning.py > added file to perform transfer learning on the model weights of : https://data2.deadtrees.earth/assets/v1/segformer_b5_full_epoch_100.safetensors 

  

