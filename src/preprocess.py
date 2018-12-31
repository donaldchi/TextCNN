#! /usr/bin/env python
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.contrib import learn
from tqdm import tqdm

from data_helpers import load_data_and_labels


def load_data():
    return pd.read_csv('../data/tweets_modified_sample.csv')


def create_vocabulary(tweets):
    max_document_length = max(
        [len(x.split(" ")) for x in tweets['words'].values]
        )
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    return vocab_processor


def vectorize_features(tweets):
    vocab_processor = create_vocabulary(tweets)
    tweets_vec = np.array(
        list(vocab_processor.fit_transform(tweets['words'].values)))
    return tweets_vec, vocab_processor


def embedding_words_with_pretrained_model(vocab_dict):
    """
        Here we use a pretrained Japanese word2vec model from
        http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
    """
    model_path = '../data/entity_vector.model.bin'
    word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)

    pre_embedding = []
    pbar = tqdm(total=len(vocab_dict))
    pbar.set_description('Embedding words by pretrained model: ')
    idx = 0
    for word in vocab_dict.keys():
        idx += 1
        np.random.seed(idx)
        vec = np.random.uniform(-0.05, 0.05, 200)
        try:
            vec = np.array(word2vec[word])
        except:
            pass
        pre_embedding.append(vec)
        pbar.update(1)
    pbar.close()
    pre_embedding = np.array(pre_embedding, dtype=np.float32)
    return pre_embedding


def split_data(tweets, tweets_vec, tweets_y):
    tweetid_vec = {}
    tweetid_y = {}
    for idx, tweet in tweets.iterrows():
        tweetid = tweet['id']
        tweetid_vec[tweetid] = tweets_vec[idx]
        tweetid_y[tweetid] = tweets_y[idx]

    # extract samples of each class
    pn = tweets[tweets['pn'] == 1]
    p = tweets[tweets['p'] == 1]
    n = tweets[tweets['n'] == 1]
    neu = tweets[tweets['neu'] == 1]
    unk = tweets[tweets['unk'] == 1]

    # split data
    train_pn = pn.sample(frac=0.8)
    test_pn = pn.drop(train_pn.index)

    train_p = p.sample(frac=0.8)
    test_p = p.drop(train_p.index)

    train_n = n.sample(frac=0.8)
    test_n = n.drop(train_n.index)

    train_neu = neu.sample(frac=0.8)
    test_neu = neu.drop(train_neu.index)

    train_unk = unk.sample(frac=0.8)
    test_unk = unk.drop(train_unk.index)

    train = train_pn
    train = train.append(train_p)
    train = train.append(train_n)
    train = train.append(train_neu)
    train = train.append(train_unk)

    test = test_pn
    test = test.append(test_p)
    test = test.append(test_n)
    test = test.append(test_neu)
    test = test.append(test_unk)

    train_vec = []
    train_y = []
    for _, item in train.iterrows():
        tweet_id = item['id']
        train_vec.append(tweetid_vec[tweet_id])
        train_y.append(tweetid_y[tweet_id])

    test_vec = []
    test_y = []
    for _, item in test.iterrows():
        tweet_id = item['id']
        test_vec.append(tweetid_vec[tweet_id])
        test_y.append(tweetid_y[tweet_id])
    print(len(train_vec), len(train_y), len(test_vec), len(test_y))

    return np.array(train_vec), np.array(train_y),
    np.array(test_vec), np.array(test_y)


def save_data(
        pre_embedding, vocab_processor, vocab_dict,
        train_vec, train_y,
        test_vec, test_y):
    pickle.dump(train_vec, open('../data/train_vec.pkl', 'wb'))
    pickle.dump(train_y, open('../data/train_y.pkl', 'wb'))
    pickle.dump(test_vec, open('../data/test_vec.pkl', 'wb'))
    pickle.dump(test_y, open('../data/test_y.pkl', 'wb'))
    pickle.dump(pre_embedding, open('../data/pre_embedding.pkl', 'wb'))
    pickle.dump(vocab_processor, open('../data/vocab_processor.pkl', 'wb'))
    pickle.dump(vocab_dict, open('../data/vocab_dict.pkl', 'wb'))


def main():
    tweets = load_data()
    tweets, tweets_y = load_data_and_labels(tweets)
    tweets_vec, vocab_processor = vectorize_features(tweets)
    vocab_dict = vocab_processor.vocabulary_._mapping

    pre_embedding = embedding_words_with_pretrained_model(vocab_dict)

    train_vec, train_y, test_vec, test_y = split_data(
        tweets, tweets_vec, tweets_y)

    save_data(
        pre_embedding, vocab_processor, vocab_dict,
        train_vec, train_y, test_vec, test_y)

if __name__ == "__main__":
    main()
