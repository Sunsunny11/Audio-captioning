# Dual Transformer for Audio captioning
Pytorch: Dual Transformer Decoder based Features Fusion Network for Automated Audio Captioning
+ Create conda environment with dependencies: **conda env create -f environment.yml -n name**
# Set up dataset
+ Run **data_prep.py** to prepare the h5py files
# Prepare evaluation tool 
+ Run **coco_caption/get_stanford_models.sh** to download the libraries necessary for evaluating the metrics.
# Run experiments
+ Set the parameters you want in **settings/settings.yaml**
+ Run experiments: **python train.py -n exp_name**
# Reinforcement learning training
+ Set settings in rl block in **settings/settings.yaml**
+ Run: **python finetune_rl.py -n exp_name**
# Citation
@INPROCEEDING{sun2023dual, <br>
title={Dual Transformer Decoder based Features Fusion Network for Automated Audio Captioning},  <br>
author={Jianyuan Sun and Xubo Liu and Xinhao Mei and Volkan Kılıç and Mark D. Plumbley and Wenwu Wang}, <br>
year={2023}, <br>
booktitle={INTERSPEECH2023}} <br>

