# TextCNN
An implementation of TextCNN from [Convolutional Neural Networks for Sentence Classification ](https://arxiv.org/abs/1408.5882).

It is a full implementation by Tensorflow including multi channels, cross-validation, utilization of pretrained embedding model.

# Usage

- train with single channel, random initialization
```
python train.py --num_epochs 10  --evaluate_every 10
```

- train with single channel, pretrained embedding initialization
```
python train.py --num_epochs 10  --evaluate_every  10 --use_pretrained_embedding
```

- train with multi channel, pretrained embedding initialization both
```
python train.py --num_epochs 10  --evaluate_every  10 --use_pretrained_embedding --use_multi_channel
```

- train with multi channel, pretrained embedding initialization and random initialization
```
python train.py --num_epochs 10  --evaluate_every  10 --use_multi_channel
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

# DataSet
- [Twitter日本語評判分析データセット](http://bigdata.naist.jp/~ysuzuki/data/twitter/)

# PreTrained model

- [日本語 Wikipedia エンティティベクトル](http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/)

# Reference
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification ](https://arxiv.org/abs/1408.5882)
