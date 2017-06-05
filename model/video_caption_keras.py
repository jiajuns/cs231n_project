import h5py
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential
from cell import LSTMDecoderCell
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input

from recurrentshop import LSTMCell
from recurrentshop import RecurrentSequential, RecurrentModel

def Seq2Seq(output_dim, output_length, batch_input_shape=None,
            input_shape=None, batch_size=None, input_dim=None, input_length=None,
            hidden_dim=None, depth=1, broadcast_state=True, unroll=False,
            stateful=False, inner_broadcast_state=True, teacher_force=False, dropout=0.):

    '''
    Seq2seq model based on [1] and [2].
    This model has the ability to transfer the encoder hidden state to the decoder's
    hidden state(specified by the broadcast_state argument). Also, in deep models
    (depth > 1), the hidden state is propogated throughout the LSTM stack(specified by
    the inner_broadcast_state argument. You can switch between [1] based model and [2]
    based model using the peek argument.(peek = True for [2], peek = False for [1]).
    When peek = True, the decoder gets a 'peek' at the context vector at every timestep.
    [1] based model:
            Encoder:
            X = Input sequence
            C = LSTM(X); The context vector
            Decoder:
    y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
    y(0) = LSTM(s0, C); C is the context vector from the encoder.
    [2] based model:
            Encoder:
            X = Input sequence
            C = LSTM(X); The context vector
            Decoder:
    y(t) = LSTM(s(t-1), y(t-1), C)
    y(0) = LSTM(s0, C, C)
    Where s is the hidden state of the LSTM (h and c), and C is the context vector
    from the encoder.
    Arguments:
    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3,
                    there will be 3 LSTMs on the enoding side and 3 LSTMs on the
                    decoding side. You can also specify depth as a tuple. For example,
                    if depth = (4, 5), 4 LSTMs will be added to the encoding side and
                    5 LSTMs will be added to the decoding side.
    broadcast_state : Specifies whether the hidden state from encoder should be
                                      transfered to the deocder.
    inner_broadcast_state : Specifies whether hidden states should be propogated
                                                    throughout the LSTM stack in deep models.
    peek : Specifies if the decoder should be able to peek at the context vector
               at every timestep.
    dropout : Dropout probability in between layers.
    '''

    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size, ) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size, ) + (input_length, ) + (input_dim, )
        else:
            shape = (batch_size, ) + (None, ) + (input_dim)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim

    encoder = RecurrentSequential(readout=True, state_sync=inner_broadcast_state,
                                  unroll=unroll, stateful=stateful,
                                  return_states=broadcast_state)
    for _ in range(depth[0]):
        encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], hidden_dim)))
        encoder.add(Dropout(dropout))

    dense1 = TimeDistributed(Dense(hidden_dim))
    dense1.supports_masking = True
    dense2 = Dense(output_dim)

    decoder = RecurrentSequential(readout='readout_only',
                                  state_sync=inner_broadcast_state, decode=True,
                                  output_length=output_length, unroll=unroll,
                                  stateful=stateful, teacher_force=teacher_force)

    for _ in range(depth[1]):
        decoder.add(Dropout(dropout, batch_input_shape=(shape[0], output_dim)))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim,
                                    batch_input_shape=(shape[0], output_dim)))

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True
    encoded_seq = dense1(_input)
    encoded_seq = encoder(encoded_seq)
    if broadcast_state:
        assert type(encoded_seq) is list
        states = encoded_seq[-2:]
        encoded_seq = encoded_seq[0]
    else:
        states = None
    encoded_seq = dense2(encoded_seq)
    inputs = [_input]
    if teacher_force:
        truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
        truth_tensor._keras_history[0].supports_masking = True
        inputs += [truth_tensor]


    decoded_seq = decoder(encoded_seq,
                          ground_truth=inputs[1] if teacher_force else None,
                          initial_readout=encoded_seq, initial_state=states)
    
    model = Model(inputs, decoded_seq)
    model.encoder = encoder
    model.decoder = decoder
    return model













