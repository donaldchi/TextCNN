#! /usr/bin/env python
import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda, 
      use_pretrained_embedding, pre_embedding, use_multi_channel):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if use_multi_channel:
                print('use_multi_channel')
                # define static channel
                self.W1 = tf.Variable(
                    pre_embedding,
                    trainable=False,
                    name="W1")

                # define non-static channel
                if use_pretrained_embedding:
                    print('use_pretrained_embedding in multichannel')
                    self.W2 = tf.Variable(
                        pre_embedding,
                        trainable=True,
                        name="W2")
                else:
                    print('not use_pretrained_embedding in multichannel')
                    self.W2 = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -0.05, 0.05),
                        trainable=True,
                        name="W2")

                self.embedded_chars1 = tf.nn.embedding_lookup(self.W1, self.input_x)
                self.embedded_chars2 = tf.nn.embedding_lookup(self.W2, self.input_x)
                self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
                self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)
                self.embedded_chars_expanded = tf.concat([self.embedded_chars_expanded1, self.embedded_chars_expanded2], 3)
            else:
                if use_pretrained_embedding:
                    print('use_pretrained_embedding in single channel')
                    self.W = tf.Variable(
                        pre_embedding,
                        name="W")
                else:
                    print('not use_pretrained_embedding in single channel')
                    self.W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -0.05, 0.05),
                        name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for _, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                if use_multi_channel:
                    filter_shape = [filter_size, embedding_size, 2, num_filters]
                else:
                    filter_shape = [filter_size, embedding_size, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, # signal
                    W,  # kernel
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.l2_loss_tmp = l2_loss
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def build_model(model_config):
    model = TextCNN(
        sequence_length=model_config['sequence_length'],
        num_classes=model_config['num_classes'],
        vocab_size=model_config['vocab_size'],
        embedding_size=model_config['embedding_size'],
        filter_sizes=model_config['filter_sizes'],
        num_filters=model_config['num_filters'],
        l2_reg_lambda=model_config['l2_reg_lambda'],
        use_pretrained_embedding=model_config['use_pretrained_embedding'],
        pre_embedding=model_config['pre_embedding'],
        use_multi_channel=model_config['use_multi_channel']
    )

    return model


def load_model(checkpoint_dir):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(
                checkpoint_file))
            saver.restore(sess, checkpoint_file)

            return sess
