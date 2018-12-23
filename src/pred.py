#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import pickle
import pandas as pd


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 1000)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_unknown", False, "Evaluate on unknown data")
tf.flags.DEFINE_string("checkpoint_number", "", "created after eatch run")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_unknown:
    x_test = pickle.load(open('../data/unknown_sample.pkl', 'rb'))
    num_classes = 5
    y_test = np.random.rand(len(x_test), num_classes)
else:
    x_test = pickle.load(open('../data/test_vec.pkl', 'rb'))
    y_test = pickle.load(open('../data/test_y.pkl', 'rb'))

# Map data into vocabulary
FLAGS.checkpoint_dir = 'runs/{}/checkpoints'.format(FLAGS.checkpoint_number)
vocab_dir = 'runs/{}'.format(FLAGS.checkpoint_number)
vocab_path = os.path.join(vocab_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

print("\nEvaluating...\n", vocab_dir)

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        feed_dict = {
                input_x: x_test,
                input_y: y_test,
                dropout_keep_prob: 1.0
        }

        predictions, accuracy = sess.run([predictions, accuracy], feed_dict)

# Print accuracy if y_test is defined
print('Accuracy: ', accuracy)
print('save prediction results')
pickle.dump(predictions[0], open('../data/pred.pkl', 'wb'))
