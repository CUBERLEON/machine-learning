import tensorflow as tf


class MNISTModel(object):
    # Model Initialization
    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size

        # Model Layers

    def model_layers(self, inputs, is_training):
        reshaped_inputs = tf.reshape(
            inputs, [-1, self.input_dim, self.input_dim, 1])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=reshaped_inputs,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name='conv1')
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
            name='pool1')
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name='conv2')
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2,
            name='pool2')
        # Dense Layer
        hwc = pool2.shape.as_list()[1:]
        flattened_size = hwc[0] * hwc[1] * hwc[2]
        pool2_flat = tf.reshape(pool2, [-1, flattened_size])
        dense = tf.layers.dense(pool2_flat, 1024,
                                activation=tf.nn.relu, name='dense')
        dropout = tf.layers.dropout(dense, rate=0.4,
                                    training=is_training)
        # Logits Layer
        logits = tf.layers.dense(dropout, self.output_size, name='logits')
        return logits

    # Runs model setup for training and evaluation
    def run_model_setup(self, inputs, labels, is_training):
        logits = self.model_layers(inputs, is_training)
        self.probs = tf.nn.softmax(logits, name='probs')
        self.predictions = tf.argmax(
            self.probs, axis=-1, name='predictions')
        class_labels = tf.argmax(labels, axis=-1)
        is_correct = tf.equal(
            self.predictions, class_labels)
        is_correct_float = tf.cast(
            is_correct,
            tf.float32)
        self.accuracy = tf.reduce_mean(
            is_correct_float)
        if self.is_training:
            labels_float = tf.cast(
                labels, tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_float,
                logits=logits)
            self.loss = tf.reduce_mean(
                cross_entropy)
            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(
                self.loss, global_step=self.global_step)

    # Run model training (See the Deep Learning for Industry course)
    def run_model_training(self, input_data, input_labels, batch_size, num_epochs, ckpt_dir):
        self.global_step = tf.train.get_or_create_global_step()
        dataset = tf.data.Dataset.from_tensor_slices((input_data, input_labels))
        dataset = dataset.shuffle(len(input_data))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        inputs, labels = iterator.get_next()
        self.run_model_setup(inputs, labels, True)

        log_vals = {'loss': self.loss, 'step': self.global_step}
        logging_hook = tf.train.LoggingTensorHook(
            log_vals, every_n_iter=50)
        tf.logging.set_verbosity(tf.logging.INFO)
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ckpt_dir,
                hooks=[tf.train.NanTensorHook(self.loss), logging_hook]) as sess:
            while not sess.should_stop():
                sess.run(self.train_op)

    # Run model evaluation
    def run_model_evaluation(self, input_data, input_labels, batch_size, ckpt_dir):
        dataset = tf.data.Dataset.from_tensor_slices((input_data, input_labels))
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        inputs, labels = iterator.get_next()
        self.run_model_setup(inputs, labels, False)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)
                acc = sess.run(self.accuracy)
                print('Accuracy: {:.3f}'.format(acc))
