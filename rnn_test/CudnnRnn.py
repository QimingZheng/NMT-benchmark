import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

CUDNN_LSTM = cudnn_rnn_ops.CUDNN_LSTM
CUDNN_LSTM_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_LSTM_PARAMS_PER_LAYER
CUDNN_RNN_UNIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
CUDNN_RNN_BIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION


class CudnnRNNModel(object):
    def __init__(
            self,
            inputs,
            rnn_mode,
            num_layers,
            num_units,
            input_size,
            initial_state=None,
            direction=CUDNN_RNN_UNIDIRECTION,
            dropout=0.,
            dtype=dtypes.float32,
            training=False,
            seed=None,
            kernel_initializer=None,
            bias_initializer=None,
    ):
        if rnn_mode == "cudnn_lstm":
            model_fn = cudnn_rnn.CudnnLSTM
        else:
            # (TODO) support other cudnn RNN ops.
            raise NotImplementedError(
                "Invalid rnn_mode: %s. Not implemented yet." % rnn_mode)

        if initial_state is not None:
            assert isinstance(initial_state, tuple)

        self._initial_state = initial_state

        self._rnn = model_fn(
            num_layers,
            num_units,
            direction=direction,
            dropout=dropout,
            dtype=dtype,
            seed=seed,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )

        # parameter passed to biuld is the shape of input Tensor.
        self._rnn.build([None, None, input_size])

        # self._outputs is a tensor of shape:
        # [seq_len, batch_size, num_directions * num_units]
        # self._output_state is a tensor of shape:
        # [num_layers * num_dirs, batch_size, num_units]

        self._outputs, self._output_state = self._rnn(
            inputs, initial_state=self._initial_state, training=training)

    @property
    def inputs(self):
        return self._inputs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def outputs(self):
        return self._outputs

    @property
    def output_state(self):
        return self._output_state

    @property
    def rnn(self):
        return self._rnn

    @property
    def total_sum(self):
        return self._AddUp(self.outputs, self.output_state)
