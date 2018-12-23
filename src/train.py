#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import feather
import pickle
import pandas as pd
from sklearn.model_selection import KFold

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string('data_dir', '../data/', "Specify data directory (default: ../data/)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_bool('use_k_fold', False, "Use k-fold cross-validation while training (default: False)")
tf.flags.DEFINE_bool('use_multi_channel', False, "Use multi channels (default: True)")
tf.flags.DEFINE_bool('use_pretrained_embedding', False, "Use pretrained embedding models, ex. word2vec (default: True)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train(x, y, x_test, y_test, vocab_processor, pre_embedding):
    # Training
    # ==================================================
    print('filter_size: ', FLAGS.filter_sizes)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x.shape[1],
                num_classes=y.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                use_pretrained_embedding=FLAGS.use_pretrained_embedding,
                pre_embedding=pre_embedding,
                use_multi_channel=FLAGS.use_multi_channel
                )

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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
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
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
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
            use_k_fold = FLAGS.use_k_fold
            if use_k_fold:
                kf = KFold(n_splits=10)
                fold_size = kf.get_n_splits() - 1
                for train_idx, dev_idx in kf.split(x, y):
                    x_train, x_dev = x[train_idx], x[dev_idx]
                    y_train, y_dev = y[train_idx], y[dev_idx]

                    # Generate batches
                    batches = data_helpers.batch_iter(
                        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                    # Training loop. For each batch...
                    train_with_batch(saver, batches, x_dev, y_dev, fold_size, fold_size, checkpoint_prefix)
            else:
                batches = data_helpers.batch_iter(list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)
                train_with_batch(saver, batches, x_test, y_test, FLAGS.evaluate_every, FLAGS.checkpoint_every, checkpoint_prefix)

                # save predict results
                feed_dict = {
                    cnn.input_x: x_test,
                    cnn.input_y: y_test,
                    cnn.dropout_keep_prob: 1.0
                }

            predictions = sess.run([cnn.predictions], feed_dict)
            predictions, scores, input_y, W = sess.run([cnn.predictions, cnn.scores, cnn.input_y, cnn.W], feed_dict)
            pickle.dump(scores, open(os.path.join(checkpoint_dir, 'scores.pkl'), 'wb'))
            pickle.dump(predictions, open(os.path.join(checkpoint_dir, 'predictions.pkl'), 'wb'))
            pickle.dump(input_y, open(os.path.join(checkpoint_dir, 'input_y.pkl'), 'wb'))
            pickle.dump(input_y, open(os.path.join(checkpoint_dir, 'W.pkl'), 'wb'))

            # save model
            current_step = tf.train.global_step(sess, global_step)
            print('checkpoint_prefix: ', checkpoint_prefix)
            print('current_step:', current_step)
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    if FLAGS.use_pretrained_embedding or FLAGS.use_multi_channel:
        pre_embedding = pickle.load(open('../data/pre_embedding.pkl', 'rb'))
    else:
        pre_embedding = None
    # load data for training
    x_train = pickle.load(
        open(os.path.join(FLAGS.data_dir, 'train_vec.pkl'), 'rb'))
    y_train = pickle.load(
        open(os.path.join(FLAGS.data_dir, 'train_y.pkl'), 'rb'))
    x_test = pickle.load(
        open(os.path.join(FLAGS.data_dir, 'test_vec.pkl'), 'rb'))
    y_test = pickle.load(
        open(os.path.join(FLAGS.data_dir, 'test_y.pkl'), 'rb'))
    vocab_processor = pickle.load(
        open(os.path.join(FLAGS.data_dir, 'vocab_processor.pkl'), 'rb'))

    checkpoint_dir = train(x_train, y_train, x_test, y_test, vocab_processor, pre_embedding)


if __name__ == '__main__':
    tf.app.run()
