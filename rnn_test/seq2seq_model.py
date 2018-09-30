import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes


class seq2seq_model(object):
    def __init__(self, mode, tgt_bos_id, tgt_eos_id):
        self.mode = mode
        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id

    def build_model(
            self,
            source,
            target_input,
            target_output,
            source_sequence_length,
            target_sequence_length,
            hparams,
            dtype=tf.float32,
    ):
        self.output_layer = Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")

        encoder_outputs, encoder_state = self._build_encoder(
            hparams, source, source_sequence_length, dtype)

        logits, final_context_state = self._build_decoder(
            hparams,
            encoder_outputs,
            encoder_state,
            source_sequence_length,
            target_input,
            target_sequence_length,
            dtype,
        )

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            loss = self._compute_loss(logits, target_output,
                                      target_sequence_length,
                                      hparams.time_major)
        else:
            loss = None

        return logits, loss, final_context_state

    def _init_embeddings(self, input, embed_name, embedding_dim, vocab_size,
                         dtype):
        embed_var = tf.get_variable(
            embed_name, [vocab_size, embedding_dim], dtype=dtype)
        return tf.nn.embedding_lookup(embed_var, input)

    def _single_cell(self, unit_type, num_units, forget_bias, dropout, mode):
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        # Cell Type
        if unit_type == "lstm":
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units, forget_bias=forget_bias)
        elif unit_type == "gru":
            single_cell = tf.contrib.rnn.GRUCell(num_units)
        elif unit_type == "layer_norm_lstm":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units, forget_bias=forget_bias, layer_norm=True)
        elif unit_type == "nas":
            single_cell = tf.contrib.rnn.NASCell(num_units)
        else:
            raise ValueError("Unknown unit type %s!" % unit_type)

        # Dropout (= 1 - keep_prob)
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))

        return single_cell

    def _build_rnn_cell(self, num_layers, unit_type, num_units, forget_bias,
                        dropout, mode):
        cell_list = []
        for i in range(num_layers):
            cell_list.append(
                self._single_cell(
                    unit_type=unit_type,
                    num_units=num_units,
                    forget_bias=forget_bias,
                    dropout=dropout,
                    mode=mode,
                ))
        if num_layers == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_bidirectional_rnn(
            self,
            inputs,
            num_layers,
            unit_type,
            num_units,
            forget_bias,
            dropout,
            mode,
            sequence_length,
            dtype,
            time_major,
    ):
        # Construct forward and backward cells
        fw_cell = self._build_rnn_cell(num_layers, unit_type, num_units,
                                       forget_bias, dropout, mode)
        bw_cell = self._build_rnn_cell(num_layers, unit_type, num_units,
                                       forget_bias, dropout, mode)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=time_major,
            swap_memory=True,
        )

        return tf.concat(bi_outputs, -1), bi_state

    def _build_encoder(self, hparams, source, source_sequence_length, dtype):
        num_layers = hparams.num_encoder_layers

        if hparams.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            encoder_emb_inp = self._init_embeddings(
                source,
                "src_embedding",
                hparams.embedding_dim,
                hparams.src_vocab_size,
                dtype,
            )

            # Encoder_outputs: [batch_size, max_time, num_units]
            if hparams.encoder_type == "cudnn_lstm":
                raise (
                    "==========cudnn_lstm is not support in this seq2seq model implementation!========="
                )

            elif hparams.encoder_type == "uni":
                cell = self._build_rnn_cell(
                    num_layers=hparams.num_encoder_layers,
                    unit_type=hparams.unit_type,
                    num_units=hparams.num_units,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.dropout,
                    mode=self.mode,
                )

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=source_sequence_length,
                    time_major=hparams.time_major,
                    swap_memory=True,
                )

            elif hparams.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)

                encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
                    inputs=encoder_emb_inp,
                    num_layers=num_bi_layers,
                    unit_type=hparams.unit_type,
                    num_units=hparams.num_units,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.dropout,
                    mode=self.mode,
                    sequence_length=source_sequence_length,
                    dtype=dtype,
                    time_major=hparams.time_major,
                )

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(
                            bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(
                            bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError(
                    "Unknown encoder_type %s" % hparams.encoder_type)
        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state):
        cell = self._build_rnn_cell(
            num_layers=hparams.num_decoder_layers,
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            mode=self.mode,
        )

        if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
            # TODO(caoying): not implemented yet.
            raise NotImplementedError("To be implemented")
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state

    def _build_decoder(
            self,
            hparams,
            encoder_outputs,
            encoder_state,
            source_sequence_length,
            target_input,
            target_sequence_length,
            dtype,
    ):

        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                if hparams.time_major:
                    target_input = tf.transpose(target_input)

                decoder_emb_inp = self._init_embeddings(
                    target_input,
                    "tgt_embedding",
                    hparams.embedding_dim,
                    hparams.tgt_vocab_size,
                    dtype,
                )

            if hparams.encoder_type == "cudnn_lstm":
                raise (
                    "==========cudnn_lstm is not support in this seq2seq model implementation!=========="
                )
            else:
                start_tokens = tf.ones([
                    hparams.batch_size,
                ], tf.int32) * self.tgt_bos_id
                end_token = self.tgt_eos_id
                embedding = tf.get_variable("embedding", [
                    hparams.tgt_vocab_size,
                    hparams.embedding_dim,
                ])
                # Helper
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding,
                    start_tokens=start_tokens,
                    end_token=end_token)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, decoding_helper, decoder_initial_state)

                # Dynamic decoding
                (outputs, final_context_state,
                 _) = tf.contrib.seq2seq.dynamic_decode(
                     decoder,
                     output_time_major=hparams.time_major,
                     swap_memory=True,
                     scope=decoder_scope,
                 )
                logits = self.output_layer(outputs.rnn_output)

        return logits, final_context_state

    def _get_max_time(self, tensor, time_major):
        time_axis = 0 if time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def _compute_loss(self, logits, target_output, target_sequence_length,
                      time_major):
        if time_major:
            target_output = tf.transpose(target_output)

        max_time = self._get_max_time(target_output, time_major)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            target_sequence_length, max_time, dtype=logits.dtype)
        if time_major:
            target_weights = tf.transpose(target_weights)
        return tf.reduce_sum(crossent * target_weights)
