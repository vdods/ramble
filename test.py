# TODO: don't use from/import; preserve the module namespaces
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import LSTM

import numpy as np
import ramble
import sys
import time

def generate_model (alphabet_size):
    hidden_layer_size = 32
    dropout = 0.2

    model = Sequential()
    model.add(LSTM(alphabet_size, hidden_layer_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer_size, hidden_layer_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(TimeDistributedDense(hidden_layer_size, alphabet_size))
    model.add(Activation('softmax'))

    sys.stdout.write('Compiling model...\n')
    start_time = time.time()
    # TODO: add loss and optimizer as parameters.
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    return model

def generate_io_tensor (symbol_sequences, alphabet, alphabetical_symbol_index, **kwargs):
    tensor_dtype                = kwargs.get('tensor_dtype', np.bool)

    symbol_sequence_count = len(symbol_sequences)
    assert symbol_sequence_count > 0, 'must supply a positive number of sequences.'
    symbol_sequence_size = len(symbol_sequences[0])
    assert all(symbol_sequence_size == len(symbol_sequence) for symbol_sequence in symbol_sequences)

    io_tensor_shape = (symbol_sequence_count, symbol_sequence_size, len(alphabet))
    io_tensor = np.zeros(io_tensor_shape, dtype=tensor_dtype)

    for symbol_sequence_index,symbol_sequence in enumerate(symbol_sequences):
        for timestep_index in range(symbol_sequence_size):
            io_tensor[symbol_sequence_index, timestep_index, alphabetical_symbol_index[ord(symbol_sequence[timestep_index])]] = 1

    return io_tensor

# def generate_prediction_io_tensors (symbol_sequences, alphabet, alphabetical_symbol_index, **kwargs):
#     return_output_symbol_tensor = kwargs.get('return_output_symbol_tensor', False)
#     tensor_dtype                = kwargs.get('tensor_dtype', np.bool)

#     sys.stdout.write( \
#         'Generating input{0} tensor{1}; len(alphabet) = {2}...\n' \
#         .format(' and output' if return_output_symbol_tensor else '', 's' if return_output_symbol_tensor else '', len(alphabet)) \
#     )
#     # sequence_tensor_shape = (1, len(symbol_sequence)-1, len(alphabet))
#     # sys.stdout.write('Created zero\'d input and output tensors of shape {0}...\n'.format(sequence_tensor_shape))
#     # start_time = time.time()
#     # input_sequence_tensor = np.zeros(sequence_tensor_shape, dtype=tensor_dtype)
#     # if return_output_symbol_tensor:
#     #     output_symbol_tensor = np.zeros(sequence_tensor_shape, dtype=tensor_dtype)
#     # sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

#     sys.stdout.write('Populating input tensor...\n')
#     start_time = time.time()
#     # for timestep_index in range(len(symbol_sequence)-1):
#     #     input_sequence_tensor[0, timestep_index, alphabetical_symbol_index[ord(symbol_sequence[timestep_index])]] = 1
#     input_sequence_tensor = generate_io_tensor(symbol_sequence[:-1], alphabet, alphabetical_symbol_index)
#     sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

#     if return_output_symbol_tensor:
#         sys.stdout.write('Populating output tensor...\n')
#         start_time = time.time()
#         # for timestep_index in range(len(symbol_sequence)-1):
#         #     output_symbol_tensor[0, timestep_index, alphabetical_symbol_index[ord(symbol_sequence[timestep_index+1])]] = 1
#         output_symbol_tensor = generate_io_tensor(symbol_sequence[1:], alphabet, alphabetical_symbol_index)
#         sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

#     # The return type depends on the value of return_output_symbol_tensor
#     if return_output_symbol_tensor:
#         return input_sequence_tensor,output_symbol_tensor
#     else:
#         return input_sequence_tensor

def fit_model_on_symbol_sequences (model, symbol_sequences, alphabet, alphabetical_symbol_index, **kwargs):
    io_tensor = generate_io_tensor(symbol_sequences, alphabet, alphabetical_symbol_index)
    return model.fit(io_tensor[:,:-1,:], io_tensor[:,1:,:], **kwargs)

def generate_symbol_sequences_from_io_tensor (io_tensor, alphabet, alphabetical_symbol_index):
    retval = []
    for sample_index in range(io_tensor.shape[0]):
        symbol_sequence = ''
        for timestep_index in range(io_tensor.shape[1]):
            symbol_sequence += alphabet[np.argmax(io_tensor[sample_index,timestep_index])]
        retval.append(symbol_sequence)
    return retval

def generate_predictions (model, symbol_sequences, alphabet, alphabetical_symbol_index):
    # sys.stdout.write('Generating prediction for symbol_sequence "{0}"...\n'.format(symbol_sequence))
    X = generate_io_tensor(symbol_sequences, alphabet, alphabetical_symbol_index)
    Y = model.predict(X, verbose=1)
    return generate_symbol_sequences_from_io_tensor(Y, alphabet, alphabetical_symbol_index)

def load_training_data ():
    training_data_filename = 'marktwain.txt'
    # Read in the training data file and generate the alphabet of recognizable chars.
    with open(training_data_filename, 'r') as f:
        sys.stdout.write('Attempting to read training data file "{0}"...\n'.format(training_data_filename))
        start_time = time.time()
        training_data = f.read()
        sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    # TODO: Figure out if alphabet [size] should be an explicit parameter of the RNN.
    training_data = training_data.lower() # Just for simplicity.
    alphabet = ramble.generate_alphabet(training_data)
    alphabetical_symbol_index = ramble.generate_alphabetical_symbol_index(alphabet)
    sys.stdout.write('Alphabet has {0} symbols and is "{1}".\n'.format(len(alphabet), alphabet))
    sys.stdout.write('Alphabetical symbol index: {0}.\n'.format(alphabetical_symbol_index))

    return training_data,alphabet,alphabetical_symbol_index

def generate_sequences (training_data, sequence_size, sequence_count):
    return [training_data[((len(training_data)-sequence_size)*i/sequence_count):((len(training_data)-sequence_size)*i/sequence_count)+sequence_size] for i in range(sequence_count)]

def generate_ramble (model, alphabet, alphabetical_symbol_index, seed_sequence, ramble_size):
    import copy
    current_sequence = copy.copy(seed_sequence)
    for _ in range(len(seed_sequence), ramble_size):
        current_io_tensor = generate_io_tensor([current_sequence], alphabet, alphabetical_symbol_index)
        next_io_tensor = model.predict(current_io_tensor)
        next_sequence = generate_symbol_sequences_from_io_tensor(next_io_tensor, alphabet, alphabetical_symbol_index)[0]
        sys.stdout.write('from "{0}" got "{1}".\n'.format(current_sequence, next_sequence))
        current_sequence += next_sequence[-1]
    return current_sequence

if __name__ == '__main__':
    create_model()

