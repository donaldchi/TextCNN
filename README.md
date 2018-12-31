# TextCNN
An implementation of TextCNN from [Convolutional Neural Networks for Sentence Classification ](https://arxiv.org/abs/1408.5882).

It is a full implementation by Tensorflow including multi channels, cross-validation, utilization of pretrained embedding model.

# Usage

- train with single channel, random initialization
```
python train.py --num_epochs 1  --evaluate_every  10 ../data/ ../model/
```

- train with single channel, pretrained embedding initialization
```
python train.py --num_epochs 1  --evaluate_every  10 --use_pretrained_embedding True ../data/ ../model/
```

- train with multi channel, pretrained embedding initialization both
```
python train.py --num_epochs 1  --evaluate_every  10 --use_pretrained_embedding True --use_multi_channel True ../data/ ../model/
```

- train with multi channel, pretrained embedding initialization and random initialization
```
python train.py --num_epochs 1  --evaluate_every  10 --use_multi_channel True ../data/ ../model/
```

- for prediction
```
python pred.py --model_dir ../model/ --model_number 1546229209  ../data/ ../result --eval_unknown 0
```

# Requirements
- Python 3.6
- TensorFlow 1.8
- pickle
- pandas
- numpy
- tqdm
- gensim
- sklearn
- beautifulsoup4

# DataSet
- [Twitter日本語評判分析データセット](http://bigdata.naist.jp/~ysuzuki/data/twitter/)

# PreTrained model

- [日本語 Wikipedia エンティティベクトル](http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/)

# 実験結果

|パラメータ <br> 組み合わせ|Accuracy(Train)|Accuracy(Test)|
|---|---:|---:|
|single channel, <br> random init|0.94|0.71|
|single channel, <br> pretrained init|0.74|0.72|
|multi channel, <br> pretrained & <br> random init|0.73|0.71|
|multi channel, <br> pretrained & <br> pretrained init|0.82|0.74|

# Reference
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification ](https://arxiv.org/abs/1408.5882)
