#! /usr/bin/env python
from tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer as CV
from tqdm import tqdm
import numpy as np


def load_data_and_labels(tweets):
    # Split by words
    wd = Tokenizer()
    cv = CV(analyzer=wd.extract_words)

    tmp = []
    tweets_y = []
    pbar = tqdm(total=len(tweets))
    pbar.set_description("Split sentences: ")
    for row, tweet in tweets.iterrows():
        pbar.update(1)
        tweet_id = tweet['id']
        tweet_genre = tweet['genre']
        pn = tweet['pn']
        p = tweet['p']
        n = tweet['n']
        neu = tweet['neu']
        unk = tweet['unk']
        content = tweet['content']

        try:
            cv.fit_transform([content])
            words = [w.lower() for w in list(cv.vocabulary_.keys())]
        except:
            print(row, tweet_id, content)
            words = ['<unk>']
        words = ' '.join(words)
        tmp.append(words)

        tweets_y.append([pn, p, n, neu, unk])

    tweets['words'] = tmp
    pbar.close()

    return tweets, tweets_y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
