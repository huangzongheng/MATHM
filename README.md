# Modality-Aware Triplet Hard Mining for Zero-Shot Sketch-Based Image Retrieval (and a strong metric learning baseline)

This project is our implementation of Modality-Aware Triplet Hard Mining (MATHM) for zero-shot sketch-based image retrieval.
More detailed descriptions and experimental results could be found in the [paper](https://arxiv.org/abs/1904.03208#).
![framework](utils/images/fig2.png)

If you find this project helpful, please consider citing our paper.
```
@article{liu2019semantic,
  author    = {Liu, Qing and Xie, Lingxi and Wang, Huiyu and Yuille, Alan},
  title     = {Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval},
  journal   = {arXiv preprint arXiv:1904.03208},
  year      = {2019},
}
```

## install dependencies
```
pip install -r requirements.txt 
```

## prepare data
Download the resized TUBerlin Ext and Sketchy Ext dataset from 
[google drive](https://drive.google.com/file/d/17hd3khIX5M2LzGK5gPOCQk7BYnUbvZOh/view?usp=sharing) or 
[baidu netdisk](https://pan.baidu.com/s/1oGwoYNf5jCCN_HtbwUt80Q) (the extraction code is 6160).

Put the unzipped folder to the same directory of this project.

## Training
Train model with our baseline method:
```
python train.py -a cse_resnet50 -d [tuberlin/sketchy/sketchy2] \
                --loss basic 
```
Train model with MATHM:
```
python train.py -a cse_resnet50 -d [tuberlin/sketchy/sketchy2] \
                --loss mathm 
```

## Testing
Evaluate our model on (TUBerlin/Sketchy) Ext with 512-d real features:
```
python test.py -a cse_resnet50 -d [tuberlin/sketchy/sketchy2]  --precision --recompute --num-hashing 512 \
               --resume-dir [path to the folder containing pre-trained model]
```

Evaluate our model on (TUBerlin/Sketchy) Ext with 64-d binary features:
```
python test.py -a cse_resnet50 -d [tuberlin/sketchy/sketchy2]  --precision --recompute --itq --num-q-hashing 64 \
               --resume-dir [path to the folder containing pre-trained model]
```

Visualize the rank list of retrieved images:
```
python test.py -a cse_resnet50 -d [tuberlin/sketchy/sketchy2]  --recompute --visualize \
               --resume-dir [path to the folder containing pre-trained model]
```


## Pre-trained model
Our trained models can be downloaded from 
[google drive](https://drive.google.com/file/d/1z0JRDRQkX9gAOlOPoDbu5qTzYGNt3kIj/view?usp=sharing) or 
[baidu netdisk](https://pan.baidu.com/s/1gF7wyxYIJwF1Sa1ZfKoNjw) (the extraction code is go78).

