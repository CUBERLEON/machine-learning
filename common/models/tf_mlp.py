import tensorflow as tf


class MLPModel(object):
    # Model Initialization
    def __init__(self, input_size, output_size):
        self.output_size = output_size
        self.inputs = tf.placeholder(tf.float32,
                                     shape=(None, input_size), name='inputs')
        self.labels = tf.placeholder(tf.int32,
                                     shape=(None, output_size), name='labels')
        self.sess = tf.Session()

    # Model Layers
    def model_layers(self):
        hidden1 = tf.layers.dense(self.inputs,
                                  5, activation=tf.nn.relu, name='hidden1')
        hidden2 = tf.layers.dense(hidden1,
                                  5, activation=tf.nn.relu, name='hidden2')
        logits = tf.layers.dense(hidden2,
                                 self.output_size, name='logits')
        return logits

    # Runs model setup for training and evaluation
    def run_model_setup(self, is_training):
        logits = self.model_layers()
        self.probs = tf.nn.softmax(logits)
        class_preds = tf.argmax(
            self.probs, axis=-1)
        self.predictions = class_preds
        class_labels = tf.argmax(
            self.labels, axis=-1)
        is_correct = tf.equal(
            self.predictions, class_labels)
        is_correct_float = tf.cast(
            is_correct,
            tf.float32)
        self.accuracy = tf.reduce_mean(
            is_correct_float)
        if is_training:
            labels_float = tf.cast(
                self.labels, tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_float,
                logits=logits)
            self.loss = tf.reduce_mean(
                cross_entropy)
            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(self.loss)
