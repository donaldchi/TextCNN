#! /usr/bin/env python
import argparse
import os
import pickle

import tensorflow as tf
from tensorflow.contrib import learn
from model_config import PRED_CONFIG
from text_cnn import load_model
import numpy as np


def load_data(eval_unknown, data_dir, model_dir, model_number):
    # CHANGE THIS: Load data. Load your own data here
    if eval_unknown:
        x_test = pickle.load(
            open('{}/unknown_sample.pkl'.format(data_dir), 'rb'))
        num_classes = 5
        y_test = np.random.rand(len(x_test), num_classes)
    else:
        x_test = pickle.load(open('{}/test_vec.pkl'.format(data_dir), 'rb'))
        y_test = pickle.load(open('{}/test_y.pkl'.format(data_dir), 'rb'))

    # Map data into vocabulary
    # vocab_processor is needed while preprocess unknonw text data
    # preprocess: convert text to vector
    vocab_dir = '{}/{}'.format(model_dir, model_number)
    vocab_path = os.path.join(vocab_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
        vocab_path)

    return x_test, y_test, vocab_processor


def prediction(model, x_test, y_test):
    input_x = model.graph.get_operation_by_name("input_x").outputs[0]
    input_y = model.graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = model.graph.get_operation_by_name(
        "dropout_keep_prob").outputs[0]

    # Tensors we want to evaluate
    predictions = model.graph.get_operation_by_name(
        "output/predictions").outputs[0]
    accuracy = model.graph.get_operation_by_name(
        "accuracy/accuracy").outputs[0]

    feed_dict = {
            input_x: x_test,
            input_y: y_test,
            dropout_keep_prob: 1.0
    }

    predictions, accuracy = model.run([predictions, accuracy], feed_dict)
    return predictions[0], accuracy


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('output_dir')

    for k, v in PRED_CONFIG.items():
        arg_type = type(v)
        parser.add_argument('--' + k, default=v, type=arg_type)

    args = parser.parse_args()
    config = dict()

    # Make the model configuration modifiable from the command line
    for k in PRED_CONFIG.keys():
        config[k] = getattr(args, k)

    return args, config


def main():
    args, config = parse_command_line()
    x_test, y_test, vocab_processor = load_data(
        args.eval_unknown, args.data_dir, args.model_dir, args.model_number)

    model_dir = '{}/{}/checkpoints'.format(args.model_dir, args.model_number)
    model = load_model(model_dir)

    pred, accuracy = prediction(model, x_test, y_test)

    # Print accuracy if y_test is defined
    if not config["eval_unknown"]:
        print('Accuracy: ', accuracy)
    print('save prediction results')
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    pickle.dump(pred, open('{}/pred.pkl'.format(args.output_dir), 'wb'))


if __name__ == "__main__":
    main()
