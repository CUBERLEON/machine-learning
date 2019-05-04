import tensorflow as tf


# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

    # Converts a sequence into an input, target pair
    def get_input_target_sequence(self, sequence):
        seq_len = len(sequence)
        if seq_len >= self.max_length:
            input_sequence = sequence[:self.max_length - 1]
            target_sequence = sequence[1:self.max_length]
        else:
            padding_amount = self.max_length - seq_len
            padding = [0 for i in range(padding_amount)]
            input_sequence = sequence[:-1] + padding
            target_sequence = sequence[1:] + padding
        return input_sequence, target_sequence

    # Create a cell for the LSTM
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    # Stack multiple layers for the LSTM
    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell

    # DO NOT MODIFY
    def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size ** 0.25)
        initial_bounds = 0.5 / embedding_dim
        initializer = tf.random_uniform(
            [self.vocab_size, embedding_dim],
            minval=-initial_bounds,
            maxval=initial_bounds)
        self.input_embedding_matrix = tf.get_variable('input_embedding_matrix',
                                                      initializer=initializer)
        input_embeddings = tf.nn.embedding_lookup(self.input_embedding_matrix, input_sequences)
        return input_embeddings

    # Run the LSTM on the input sequences
    def run_lstm(self, input_sequences, is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        binary_sequences = tf.sign(input_sequences)
        sequence_lengths = tf.reduce_sum(binary_sequences, axis=1)
        lstm_outputs, _ = tf.nn.dynamic_rnn(
            cell,
            input_embeddings,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        return lstm_outputs, binary_sequences

    # Calculate the overall loss for a language model training step
    def calculate_loss(self, lstm_outputs, binary_sequences, output_sequences):
        logits = tf.layers.dense(lstm_outputs, self.vocab_size)
        batch_sequence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_sequences, logits=logits)
        unpadded_loss = batch_sequence_loss * tf.cast(binary_sequences, tf.float32)
        overall_loss = tf.reduce_sum(unpadded_loss)
        return overall_loss

    # DO NOT MODIFY
    # See the Deep Learning for Industry course for details
    def restore_and_run(self, run_vals, ckpt_dir):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            return sess.run(run_vals)

    # Generate next word ID prediction
    def generate_next_word_id(self, input_sequences, batch_size, ckpt_dir):
        lstm_outputs, binary_sequences = self.run_lstm(input_sequences, False)
        logits = tf.layers.dense(lstm_outputs, self.vocab_size)
        probabilities = tf.nn.softmax(logits, axis=-1)
        top_ids = tf.argmax(probabilities, axis=-1)
        row_indices = tf.range(batch_size)
        final_indexes = tf.reduce_sum(binary_sequences, axis=1) - 1
        gather_indices = tf.transpose([row_indices, final_indexes])
        final_id_predictions = tf.gather_nd(top_ids, gather_indices)
        return self.restore_and_run(final_id_predictions, ckpt_dir)

    # See the Model Execution Lab for more details on tf.Example
    def sequence_to_example(self, sequence):
        input_sequence, target_sequence = self.get_input_target_sequence(sequence)
        input_feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=input_sequence))
        target_feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=target_sequence))
        feature_dict = {
            'input_ids': input_feature,
            'target_ids': target_feature
        }
        features_obj = tf.train.Features(feature=feature_dict)
        return tf.train.Example(features=features_obj)

    # Converts a list of texts into a TFRecords file
    def write_data_files(self, texts, tfrecords_file):
        writer = tf.python_io.TFRecordWriter(tfrecords_file)
        for sequence in self.tokenize_text_corpus(texts):
            if len(sequence) > 1:
                example = self.sequence_to_example(sequence)
                writer.write(example.SerializeToString())
        writer.close()

    # See the Model Execution Lab for more details on converting raw data into an efficient data pipeline
    def dataset_from_examples(self, tfrecord_files, batch_size, buffer_size, num_epochs):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        example_spec = {
            'input_ids': tf.FixedLenFeature((self.max_length - 1,), tf.int64),
            'target_ids': tf.FixedLenFeature((self.max_length - 1,), tf.int64)
        }

        def _parse_fn(example_bytes):
            parsed_features = tf.parse_single_example(example_bytes, example_spec)
            input_ids = parsed_features['input_ids']
            target_ids = parsed_features['target_ids']
            return input_ids, target_ids

        dataset = dataset.map(_parse_fn).shuffle(buffer_size)
        return dataset.repeat(num_epochs).batch(batch_size)

    # See the Model Execution Lab for more details on efficient model training
    def run_training(self, tfrecord_files, ckpt_dir, batch_size, buffer_size, num_epochs=None):
        dataset = self.dataset_from_examples(tfrecord_files, batch_size, buffer_size, num_epochs)
        iterator = dataset.make_one_shot_iterator()
        input_sequences, output_sequences = iterator.get_next()
        lstm_outputs, binary_sequences = self.run_lstm(input_sequences, True)
        loss = self.calculate_loss(lstm_outputs, binary_sequences, output_sequences)
        global_step = tf.train.get_or_create_global_step()
        adam = tf.train.AdamOptimizer()
        train_op = adam.minimize(loss, global_step=global_step)
        log_vals = {'loss': loss, 'step': global_step}
        logging_hook = tf.train.LoggingTensorHook(
            log_vals, every_n_secs=60)
        nan_hook = tf.train.NanTensorHook(loss)
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=ckpt_dir,
                hooks=[logging_hook, nan_hook]) as sess:
            while not sess.should_stop():
                sess.run(train_op)
