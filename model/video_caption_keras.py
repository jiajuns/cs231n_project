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

from recurrentshop import LSTMCell
from recurrentshop import RecurrentSequential, RecurrentModel

def sequence_2_sequence(output_units, input_size, frame_num, sentence_len):
    encoder = RecurrentSequential()
    lstm_cell = LSTMCell(output_units, input_dim=input_size)

    input = Input((5,))
    state1_tm1 = Input((10,))
    state2_tm1 = Input((10,))
    state3_tm1 = Input((10,))

    for _ in range(frame_num):
        lstm_output, state1_t, state2_t = LSTMCell(10)([input, state1_tm1, state2_tm1])
