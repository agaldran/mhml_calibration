# Multi-Head Multi-Loss Model Calibration
You have reached the repository with code to reproduce results in the paper:

```
Multi-Head Multi-Loss Model Calibration 
Adrian Galdran, Johan Verjans, Gustavo Carneiro, and Miguel A. González Ballester
Under review, fingers crossed, 2023
```
The paper is in arXiv, you can get it at this [link](https://arxiv.org/abs/2303.01099).

## What's the deal? Quick Explanation
We found that if one replaces the single prediction branch in a classifier by several branches, and then trains the model 
by minimizing a differently weighted CE loss at each branch, the resulting classifier is quite well calibrated.

If you want to use the multi-head model implementation, it is in `utils/get_model_v2.py`. 
Note that it returns a tensor of shape batch_size x num_heads x num_classes during training, but if you later do `model.eval()` 
then what you get is the predictions averaged over heads, with a softmax already applied unless you specify otherwise. 

## Dependencies
You want to install the usual, `pytorch`, `torchvision`, `scikit-learn`, `scikit-image`, `tqdm`, `pandas`, `jupyter`, `tochmetrics`. 
   
## Data availability
The Kvasir dataset can be downloaded [here](https://datasets.simula.no/hyper-kvasir/), 
and the Chaoyang dataset [here](https://bupt-ai-cz.github.io/HSA-NRL/) (you need to request access). 
Automatic downloading of the kvasir data can be triggered by running:
```
sh get_data.sh
```
That script will also put everything in place and ready to train.
Train, Val, Test splits for both datasets are already specified in the `data` directory, 
but you still have to download the Chaoyang dataset manually and move it to that directory.

## Running experiments - Training
For training a single-head model you need to use the `train_other.py` script as follows:
```
python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_ls --hypar 0.05 --save_path kvasir/ls_005/cnx_f0 --seed 0
```
This trains a model with Label Smoothing, with a smoothing of 0.05, on the Kvasir dataset. 
You can choose other architectures, in the paper we evaluated `convnext`, `resnet50`, and `swin`.
You can also pick other methods for comparison. 
If you want to train a multi-head model with, say 4 heads, you would run:

```
python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 4 --save_path kvasir/4hs/cnx_f0 --seed 0
```

All the experiments in the paper that used, for example, a convnext architecture and the Kvasir dataset, can be run at once using:
```
sh train_PMH_convnext_kvasir.sh
```
There you have examples of all the options required to reproduce that part of the experiments the paper. 
If you really want to retrain everything, you would need to replicate that file but using the  `resnet50`, and `swin` architectures, 
and then do the same for the Chaoyang dataset.


## Running experiments - Testing
Once you have trained all models on a particular dataset*, you can  analyze results using `jupyter notebook` (sorry about that).
Anyway, you would need to open `train_PMH_convnext_chaoyang.ipynb` to check performance. 
In addition to computing and printing out everything, it should also give you an enormous string that you could then 
copy-paste over into a latex editor to get something like [this](https://quicklatex.com/cache3/33/ql_1d7df236d8315fc1e60966c36f0cd933_l3.png).

*: We are doing five fold so it will take a while, think around 10-15 minutes per model, and there are lots of methods too.
