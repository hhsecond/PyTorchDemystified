import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import pickle

from util import get_data, get_accuracy

en, fr = get_data()


def model_inputs():
    input_ = tf.placeholder(tf.int32, [None, None], name="input")
    target_ = tf.placeholder(tf.int32, [None, None], name="target")
    learn_rate_ = tf.placeholder(tf.float32, name="learn_rate")
    keep_prob_ = tf.placeholder(tf.float32, name="keep_prob")
    targ_seq_len_ = tf.placeholder(tf.int32, [None], name="target_sequence_length")
    max_targ_seq_len_ = tf.reduce_max(targ_seq_len_, name="max_target_len")
    source_seq_len_ = tf.placeholder(tf.int32, [None], name="source_sequence_length")
    return input_, target_, learn_rate_, keep_prob_, targ_seq_len_, max_targ_seq_len_, source_seq_len_


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # Take off the last column
    sliced = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # Append a column filled with <GO>
    decoder_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), sliced], 1)
    return decoder_input


def make_lstm(size, keep_prob):
    lstm_cell = tf.contrib.rnn.LSTMCell(size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return drop


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    # Embed the input
    encoding_embed_input = tf.contrib.layers.embed_sequence(
        rnn_inputs, source_vocab_size, encoding_embedding_size)
    # Make a stacked LSTM
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [make_lstm(rnn_size, keep_prob) for i in range(num_layers)])
    # Run the LSTM
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm,
                                       encoding_embed_input,
                                       sequence_length=source_sequence_length,
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    # Define the helper
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=dec_embed_input,
        sequence_length=target_sequence_length,
        time_major=False)
    # Define the decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)
    # Run the decoder
    train_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        impute_finished=True,
        maximum_iterations=max_summary_length)
    return train_decoder_output


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    start_tokens = tf.tile(
        tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')
    # Define the helper
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        dec_embeddings,
        start_tokens,
        end_of_sequence_id)
    # Define the decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell,
        helper,
        encoder_state,
        output_layer)
    # Run the decoder
    infer_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        impute_finished=True,
        maximum_iterations=max_target_sequence_length)
    return infer_decoder_output


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    # Define the embeddings and the embedded inputs
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    # Stack LSTMs
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [make_lstm(rnn_size, keep_prob) for _ in range(num_layers)])
    # Use the Dense function from tf.python.layers.core to make a fully connected layer
    output_layer = Dense(
        target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # Run the training
    with tf.variable_scope("decode") as scope:
        dec_train = decoding_layer_train(encoder_state,
                                         stacked_lstm,
                                         dec_embed_input,
                                         target_sequence_length,
                                         max_target_sequence_length,
                                         output_layer,
                                         keep_prob)
        # Reuse the variables that we used to train it on
        scope.reuse_variables()
        dec_infer = decoding_layer_infer(encoder_state,
                                         stacked_lstm,
                                         dec_embeddings,
                                         target_vocab_to_int['<GO>'],
                                         target_vocab_to_int['<EOS>'],
                                         max_target_sequence_length,
                                         target_vocab_size,
                                         output_layer,
                                         batch_size,
                                         keep_prob)
    return dec_train, dec_infer


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    # Initialize the encoding layer
    _, enc_state = encoding_layer(
        input_data,
        rnn_size,
        num_layers,
        keep_prob,
        source_sequence_length,
        source_vocab_size,
        enc_embedding_size)
    # Process the input for the decoder
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    # Initialize the decoding layer
    training_decoder_output, inference_decoder_output = decoding_layer(
        dec_input,
        enc_state,
        target_sequence_length,
        max_target_sentence_length,
        rnn_size,
        num_layers,
        target_vocab_to_int,
        target_vocab_size,
        batch_size,
        keep_prob,
        dec_embedding_size)
    return training_decoder_output, inference_decoder_output


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    temp = [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    return temp


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


epochs = 8
batch_size = 128
rnn_size = 256
num_layers = 2
encoding_embedding_size = 128
decoding_embedding_size = 128
learning_rate = 0.001
keep_probability = 0.5
display_step = 100


max_target_sentence_length = max([len(sentence) for sentence in en.text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="target")
    lr = tf.placeholder(tf.float32, name="learn_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name="max_target_len")
    source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(en.vocab2id),
                                                   len(fr.vocab2id),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   fr.vocab2id)

    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(
        target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# Split data to training and validation sets
train_source = en.text_as_ids[batch_size:]
train_target = fr.text_as_ids[batch_size:]
valid_source = en.text_as_ids[:batch_size]
valid_target = fr.text_as_ids[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = next(
    get_batches(
        valid_source,
        valid_target,
        batch_size,
        en.vocab2id['<PAD>'],
        fr.vocab2id['<PAD>']))
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            en.vocab2id['<PAD>'],
                            fr.vocab2id['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})

            if batch_i % display_step == 0 and batch_i > 0:

                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print(
                    'Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(en.text_as_ids) // batch_size, train_acc, valid_acc, loss))

    saver = tf.train.Saver()
    saver.save(sess, 'dev')
    print('Model Trained and Saved')


with open('preprocess.p', mode='rb') as in_file:
    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = pickle.load(in_file)


def sentence_to_seq(sentence, vocab_to_int):
    text = sentence.lower().split()
    word_ids = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text]
    return word_ids


translate_sentence = 'he saw a old yellow truck .'
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('dev.meta')
    loader.restore(sess, 'dev')

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    translate_logits = sess.run(logits, {input_data: [translate_sentence] * batch_size,
                                         target_sequence_length: [len(translate_sentence) * 2] * batch_size,
                                         source_sequence_length: [len(translate_sentence)] * batch_size,
                                         keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words:  {}'.format(([target_int_to_vocab[i] for i in translate_logits])))

