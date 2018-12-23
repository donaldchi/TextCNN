#! /usr/bin/env python
import MeCab


class Tokenizer:
    INDEX_CATEGORY = 0
    INDEX_ROOT_FORM = 6
    TARGET_CATEGORIES = [
        "名詞", "固有名詞", "代名詞", "形容詞", "冠詞",
        "数詞", "動詞", "代動詞", "助動詞", "副詞", "前置詞",
        "接続詞", "感動詞", "連体詞"]

    def __init__(self, dictionary="mecabrc"):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger('-Ochasen')

    def extract_words(self, text):
        if not text:
            return []

        words = []

        node = self.tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')

            if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES:
                if features[self.INDEX_ROOT_FORM] == "*":
                    words.append(node.surface)
                else:
                    words.append(features[self.INDEX_ROOT_FORM])

            node = node.next

        return words
