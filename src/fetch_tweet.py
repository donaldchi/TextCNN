"""
twitter.com/anyuser/status/541278904204668929
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import os
from joblib import Parallel, delayed

BASE_URL = 'https://twitter.com/anyuser/status'


def load_statuses():
    tweets = pd.read_csv('../data/tweets_open.csv', header=None)
    return tweets[2].values


def fetch_tweets(tweet_statuses):
    file_name_with_path = '../data/tweets.csv'
    if os.path.exists(file_name_with_path):
        f = open(file_name_with_path, 'a')
    else:
        f = open(file_name_with_path, 'w')

    pbar = tqdm(total=len(tweet_statuses))
    pbar.set_description('Fetch tweets: ')
    class_id = 'TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text'
    count = 0
    for tweet_status in tweet_statuses:
        try:
            url = '{}/{}'.format(BASE_URL, tweet_status)
            html = requests.get(url)
            soup = BeautifulSoup(html.content, 'html.parser')
            content = soup.find('p', class_=class_id).getText()
            content = content.replace('\n', ' ').replace('\t', ' ')
            f.write('{}, {}\n'.format(tweet_status, content))
            f.flush()
        except:
            pass
        pbar.update(1)
    f.close()
    pbar.close()


def main():
    print('main')
    tweet_statuses = load_statuses()
    fetch_tweets(tweet_statuses)

if __name__ == "__main__":
    main()
