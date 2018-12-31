#! /usr/bin/env python
import argparse
import datetime
import os
import pickle
import time

import tensorflow as tf
from sklearn.model_selection import KFold

from data_helpers import batch_iter
from model_config import DEFAULT_CONFIG
from text_cnn import build_model


def _define_model_config(data, pre_embedding, config):
    return {
        "sequence_length": data['x_train'].shape[1],
        "num_classes": data['y_train'].shape[1],
        "vocab_size": len(data['vocab_processor'].vocabulary_),
        "embedding_size": config['embedding_dim'],
        "filter_sizes": list(map(int, config['filter_sizes'].split(","))),
        "num_filters": config['num_filters'],
        "l2_reg_lambda": config['l2_reg_lambda'],
        "use_pretrained_embedding": config['use_pretrained_embedding'],
        "pre_embedding": pre_embedding,
        "use_multi_channel": config['use_multi_channel']
    }


def train(data, pre_embedding, config, output_dir):
    # x, y, x_test, y_test, vocab_processor
    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config['allow_soft_placement'],
          log_device_placement=config['log_device_placement'])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model_config = _define_model_config(data, pre_embedding, config)
            cnn = build_model(model_config)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.join(output_dir, timestamp)
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config['num_checkpoints'])

            # Write vocabulary
            data['vocab_processor'].save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: config['dropout_keep_prob']
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("Train: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Test: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            def train_with_batch(saver, batches, x_dev, y_dev, evaluate_every, checkpoint_every, checkpoint_prefix):
                """
                Train model with batches
                """
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % evaluate_every == 0:
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)

            # Generate batches
            use_k_fold = config['use_k_fold']
            if use_k_fold:
                kf = KFold(n_splits=10)
                fold_size = kf.get_n_splits() - 1
                for train_idx, dev_idx in kf.split(x, y):
                    x_train = data['x_train'][train_idx]
                    x_dev = data['x_test'][dev_idx]

                    y_train = data['y_train'][train_idx]
                    y_dev = data['y_test'][dev_idx]

                    # Generate batches
                    batches = batch_iter(
                        list(zip(x_train, y_train)), config['batch_size'], config['num_epochs'])
                    # Training loop. For each batch...
                    train_with_batch(saver, batches, x_dev, y_dev, fold_size, fold_size, checkpoint_prefix)
            else:
                batches = batch_iter(
                    list(zip(data['x_train'], data['y_train'])),
                    config['batch_size'], config['num_epochs'])
                train_with_batch(
                    saver, batches, data['x_test'], data['y_test'],
                    config['evaluate_every'], config['checkpoint_every'],
                    checkpoint_prefix)
                # save predict results
                feed_dict = {
                    cnn.input_x: data['x_test'],
                    cnn.input_y: data['y_test'],
                    cnn.dropout_keep_prob: 1.0
                }

            predictions = sess.run([cnn.predictions], feed_dict)
            predictions, scores, input_y= sess.run([cnn.predictions, cnn.scores, cnn.input_y], feed_dict)
            pickle.dump(scores, open(os.path.join(checkpoint_dir, 'scores.pkl'), 'wb'))
            pickle.dump(predictions, open(os.path.join(checkpoint_dir, 'predictions.pkl'), 'wb'))
            pickle.dump(input_y, open(os.path.join(checkpoint_dir, 'input_y.pkl'), 'wb'))

            # save model
            current_step = tf.train.global_step(sess, global_step)
            print('checkpoint_prefix: ', checkpoint_prefix)
            print('current_step:', current_step)
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


def load_data(path):
    # load data for training
    x_train = pickle.load(open(os.path.join(path, 'train_vec.pkl'), 'rb'))
    y_train = pickle.load(open(os.path.join(path, 'train_y.pkl'), 'rb'))
    x_test = pickle.load(open(os.path.join(path, 'test_vec.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(path, 'test_y.pkl'), 'rb'))

    # load for computing precision / recall
    path = os.path.join(path, 'vocab_processor.pkl')
    vocab_processor = pickle.load(open(path, 'rb'))

    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'vocab_processor': vocab_processor,
    }
    return data


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--verbose', action='store_true')

    for k, v in DEFAULT_CONFIG.items():
        arg_type = type(v)
        parser.add_argument('--' + k, default=v, type=arg_type)

    args = parser.parse_args()
    config = dict()

    # Make the model configuration modifiable from the command line
    for k in DEFAULT_CONFIG.keys():
        config[k] = getattr(args, k)

    return args, config


def main(argv=None):
    args, config = parse_command_line()

    if args.verbose:
        print('Loading data ...')

    if config['use_pretrained_embedding'] or config['use_multi_channel']:
        path = os.path.join(args.data_dir, 'pre_embedding.pkl')
        pre_embedding = pickle.load(open(path, 'rb'))
    else:
        pre_embedding = None

    data = load_data(args.data_dir)

    checkpoint_dir = train(data, pre_embedding, config, args.output_dir)


if __name__ == '__main__':
    tf.app.run()
