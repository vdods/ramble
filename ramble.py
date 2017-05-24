"""
Ramble -- RNN-based compression

Design notes:

Using a character-sequence-predicting RNN, it is possible to get a rather high accuracy for
prediction of certain classes of sequential data (e.g. English text, markdown, C code, etc).
This sequence prediction can be used to convert the sequential data into another sequence whose
frequency histogram is highly skewed such that one symbol's frequency is much higher than 50%.
This skewed sequence is more conducive to high compression than the original sequence.

The program will have several modes in which in can run:
- Training mode: Given a large sample of a particular class of sequential data, this will
  train an RNN (of specified format) to predict next char(s) in sequences.  The training
  can happen in multiple sessions, and each step of training can/will be stored to disk.
  Training should record an analysis of its quality for each epoch, so that its progress
  and quality can be tracked by a human user, for example to know when to stop training.
- Debug/analysis mode: TODO (this should really just be for Ramble devs).
- Compression mode: Use a specified trained RNN to convert an input sequence (from file or
  stdin) to the more-highly-compressable sequence mentioned in the summary above.  Then
  compress this sequence with a specified compression scheme.
- Decompression mode: Exactly what it sounds like.  Note that this also requires using the
  RNN for sequential predictions.

Training file directory format:
- Because the training phase is so computationally intensive, the weights data for each intermediate
  state of the NN will be stored, so that there is a backup of the current progress if the training
  is interrupted for whatever reason.  The intermediate state data will (for now) be the same as the
  'final' training data that's used while compressing/decompressing.
- The parameters for an RNN are:
  * 'data-class-name' : string containing only hypens and lowercase alphanumeric chars (e.g. 'english-text', 'c-code', 'python-2-7', etc.)
  * 'training-data-filename' : string (the large corpus of example sequential data)
  * 'hidden-layer-count' : positive integer (the number of hidden layers in the RNN)
  * 'hidden-layer-size' : positive integer (the number of neurons in each hidden layer of the RNN)
  * 'dropout' : float between 0.0 and 1.0 (a value used in the training to prevent redundancy)
  * 'epoch-count' : positive integer (the assumption is more epochs means higher accuracy)
- The traing data filename will be deterministically generated from the RNN parameters as follows:

    ramble.{name0}:{value0},{name1}:{value1}.rnn

  where [name0, name1, ...] is the alphabetized list of parameter names above and value0, value1, ...
  are their corresponding values.
- Additionally, the results of the analysis done after each epoch will be stored so that a human
  user can track the progress and quality of the training for each epoch.  This data will be stored
  in a pickle so that it can be read back later and the entire analysis history is available for
  generating reports.  The filename will be

    ramble.{name0}:{value0},{name1}:{value1}.metadata.pickle

  Additional reports can be made, e.g. plots of accuracy, time taken per training epoch, etc.
  There will be a text summary report, generated from the analysis history after each epoch,
  which will include the following information:
  * Epoch index
  * Loss value
  * Accuracy (testing the RNN on the validation data)
  * Time taken to run this epoch of training
  * A sampling of the RNN for different 'diversity' values
  * The rank frequencies on a particular test dataset
  * The sequence length histograms for symbol classes 0 and non-0
  * The average sequence lengths for symbol classes 0 and non-0
  * Time taken to generate the rank sequence of the test dataset

TODO:
- Incorporate alphabet into .rnn file.  Perhaps also training analysis log.
- Handle non-alphabetical input symbols (project to some sentinel symbol).
- Experiment with binary sequence prediction (this would eliminate the need for an alphabet and
  would make a lot of the tensors smaller, at the expense of using longer sequences, but binary
  sequences seem more fundamental/essential).
- Understand better the usage/formatting of input/output of RNNs.  In particular, because we're
  interested in predicting contiguous sequences of chars (i.e. with temporal coherence), is it
  possible to run the RNN with sequence input and output, instead of lots of overlapping sequences
  for input?
"""

import getopt
import itertools
import numpy as np
import os
import pickle
import sys
import time

# TODO: don't use from/import; preserve the module namespaces
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def ramble_usage_string (argv):
    return 'usage: {0} [options]\n'.format(argv[0]) # TODO: full help string

# # Old version -- use generate_rank_sequence_2 instead.
# def generate_rank_sequence (model, symbol_sequence, training_sequence_size, alphabet, alphabetical_symbol_index):
#     sequence_count = len(symbol_sequence) - training_sequence_size - 1
#     assert sequence_count > 0

#     rank_sequence = []
#     sys.stdout.write('Generating prediction for {0} sequences...\n'.format(sequence_count))
#     start_time = time.time()
#     for start_index in range(sequence_count):
#         sentence = symbol_sequence[start_index : start_index + training_sequence_size]
#         actual_next_symbol = symbol_sequence[start_index+training_sequence_size]
#         assert len(sentence) == training_sequence_size

#         x = np.zeros((1, training_sequence_size, len(alphabet)))
#         for timestep_index,symbol in enumerate(sentence):
#             x[0,timestep_index,alphabetical_symbol_index[ord(symbol)]] = 1.0

#         prediction = model.predict(x, verbose=0)

#         ordered_symbol_indices = prediction.reshape(prediction.size).argsort()[::-1] # get top candidates. SLOW
#         ordered_symbols = [alphabet[symbol_index] for symbol_index in ordered_symbol_indices]
#         # TODO: handle case where there's a tie for first in a well-defined way
#         rank = ordered_symbols.index(actual_next_symbol)
#         assert 0 <= rank < len(alphabet)
#         rank_sequence.append(rank)
#     sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

#     return rank_sequence

def generate_model_io_tensors (symbol_sequence, sequence_start_indices, sequence_size, alphabet, alphabetical_symbol_index, **kwargs):
    return_output_symbol_tensor = kwargs.get('return_output_symbol_tensor', False)
    tensor_dtype                = kwargs.get('tensor_dtype', np.bool)

    sequence_count = len(sequence_start_indices)
    assert sequence_count > 1
    sys.stdout.write( \
        'Generating input{0} tensor{1}; sequence_count = {2}, sequence_size = {3}, len(alphabet) = {4}...\n' \
        .format(' and output' if return_output_symbol_tensor else '', 's' if return_output_symbol_tensor else '', sequence_count, sequence_size, len(alphabet)) \
    )
    input_sequence_tensor_shape = (sequence_count, sequence_size, len(alphabet))
    sys.stdout.write('Created zero\'d input tensor of shape {0}...\n'.format(input_sequence_tensor_shape))
    start_time = time.time()
    input_sequence_tensor = np.zeros(input_sequence_tensor_shape, dtype=tensor_dtype)
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    if return_output_symbol_tensor:
        output_sequence_tensor_shape = (sequence_count, len(alphabet))
        sys.stdout.write('Created zero\'d output tensor of shape {0}...\n'.format(input_sequence_tensor_shape))
        start_time = time.time()
        output_symbol_tensor = np.zeros(output_sequence_tensor_shape, dtype=tensor_dtype)
        sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    sys.stdout.write('Populating input tensor...\n')
    start_time = time.time()
    for start_index in sequence_start_indices:
        for timestep_index in range(sequence_size):
            input_sequence_tensor[start_index, timestep_index, alphabetical_symbol_index[ord(symbol_sequence[start_index+timestep_index])]] = 1
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    if return_output_symbol_tensor:
        sys.stdout.write('Populating output tensor...\n')
        start_time = time.time()
        for start_index in sequence_start_indices:
            output_symbol_tensor[start_index, alphabetical_symbol_index[ord(symbol_sequence[start_index+sequence_size])]] = 1
        sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    # The return type depends on the value of return_output_symbol_tensor
    if return_output_symbol_tensor:
        return input_sequence_tensor,output_symbol_tensor
    else:
        return input_sequence_tensor

def generate_predictions (model, symbol_sequence, training_sequence_size, alphabet, alphabetical_symbol_index):
    sequence_count = len(symbol_sequence)-training_sequence_size-1
    assert sequence_count > 1
    input_sequence_tensor = generate_model_io_tensors(symbol_sequence, range(sequence_count), training_sequence_size, alphabet, alphabetical_symbol_index)

    # Using a single, tensor-based predict step doesn't seem to have any speed increase on a non-GPU system.
    sys.stdout.write('Generating prediction for {0} sequences...\n'.format(sequence_count))
    start_time = time.time()
    predictions = model.predict(input_sequence_tensor, verbose=1)
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    return predictions

# This is the newer, better version (don't use generate_rank_sequence anymore)
def generate_rank_sequence_2 (model, symbol_sequence, training_sequence_size, alphabet, alphabetical_symbol_index, **kwargs):
    return_rank_likelihoods = kwargs.get('return_rank_likelihoods', False)

    rank_sequence = []
    if return_rank_likelihoods:
        rank_likelihoods = []

    sequence_count = len(symbol_sequence)-training_sequence_size-1
    assert sequence_count > 1
    predictions = generate_predictions(model, symbol_sequence, training_sequence_size, alphabet, alphabetical_symbol_index)
    assert len(predictions) == sequence_count
    for sequence_index,prediction in enumerate(predictions):
        assert len(prediction) == len(alphabet)

        actual_next_symbol_index = alphabetical_symbol_index[ord(symbol_sequence[sequence_index+training_sequence_size])]

        ranked_prediction = sorted(enumerate(prediction), reverse=True, key=(lambda x : x[1])) # x is (symbol_index, symbol_likelihood)
        ranked_symbol_indices,ranked_symbol_likelihoods = zip(*ranked_prediction)
        actual_next_symbol_rank = ranked_symbol_indices.index(actual_next_symbol_index)
        rank_sequence.append(actual_next_symbol_rank)
        if return_rank_likelihoods:
            actual_next_symbol_likelihood = ranked_symbol_likelihoods[actual_next_symbol_rank]
            rank_likelihoods.append(actual_next_symbol_likelihood)

    if return_rank_likelihoods:
        return rank_sequence,rank_likelihoods
    else:
        return rank_sequence

def compute_frequencies (integer_sequence, integer_upper_bound):
    frequencies = [0.0] * integer_upper_bound
    # Accumulate the counts of each value
    for value in integer_sequence:
        assert 0 <= value < integer_upper_bound, 'value in integer_sequence was not in the expected range [0,{0}).'.format(integer_upper_bound)
        frequencies[value] += 1.0
    # Normalize so that the sum of the frequencies is 1.0
    for i in range(integer_upper_bound):
        frequencies[i] /= len(integer_sequence)
    return frequencies

def compute_sequence_length_histograms (integer_sequence, integer_upper_bound):
    histograms = [{} for _ in range(integer_upper_bound)]

    def record_sequence_length (value, sequence_length):
        h = histograms[value]
        if sequence_length not in h:
            h[sequence_length] = 1
        else:
            h[sequence_length] += 1

    previous_value = None
    current_sequence_length = 0
    for value in integer_sequence:
        assert 0 <= value < integer_upper_bound, 'value ({0}) in integer_sequence was not in the expected range [0,{1}).'.format(value, integer_upper_bound)
        if previous_value == None:          # Initialize previous_value if necessary.
            previous_value = value

        if value == previous_value:         # Continue the current sequence.
            current_sequence_length += 1
        else:                               # Record the current sequence and reset it.
            record_sequence_length(previous_value, current_sequence_length)
            current_sequence_length = 1

        previous_value = value

    if previous_value != None:
        record_sequence_length(previous_value, current_sequence_length)

    return histograms

def compute_average_sequence_lengths (sequence_length_histograms):
    return [sum(sequence_length*count for sequence_length,count in sequence_length_histogram.iteritems()) \
            / \
            float(sum(count for count in sequence_length_histogram.values())) for sequence_length_histogram in sequence_length_histograms]

class RNNParameterModel:
    def __init__ (self, parameter_spec):
        self.names                = parameter_spec.keys()
        self.description          = {name : parameter_spec[name]['description']          for name in self.names}
        self.value_type           = {name : parameter_spec[name]['value-type']           for name in self.names}
        self.value_from_string    = {name : parameter_spec[name]['value-from-string']    for name in self.names}
        self.default_value        = {name : parameter_spec[name]['default-value']        for name in self.names}
        self.validator            = {name : parameter_spec[name]['validator']            for name in self.names}
        self.validity_description = {name : parameter_spec[name]['validity-description'] for name in self.names}

    def parse_parameter_string_dict (self, parameter_string_dict):
        retval = {}
        for name in self.names:
            if name in parameter_string_dict:
                value_string = parameter_string_dict[name]
                # print 'name,value_string = ', (name,value_string)
                assert type(value_string) == str, 'expected a str, got a {0} (value was {1})'.format(type(value_string), value_string)
                try:
                    value = self.value_from_string[name](value_string)
                except Exception as e:
                    raise Exception('Failed to parse value string "{0}" for RNN parameter "{1}" ({2}); it {3}.  Parse error was {4}.'.format(value_string, name, self.description[name], self.validity_description[name], str(e)))
                if not (type(value) == self.value_type[name] and self.validator[name](value)):
                    raise Exception('RNN parameter "{0}" ({1}) value "{2}" is invalid; it {3}.'.format(name, self.description[name], value_string, self.validity_description[name]))
                retval[name] = value
            else:
                value = self.default_value[name]
                # print 'name,default_value = ', (name,value)
                if value == None:
                    raise Exception('No value specified for required RNN parameter "{0}" ({1}); it {2}.'.format(name, self.description[name], self.validity_description[name]))
                else:
                    assert type(value) == self.value_type[name] and self.validator[name](value), 'Inconsistent/incorrectly specified RNN parameter model for name {0}'.format(name)
                    retval[name] = value

        # If some of the parameter_string_dict keys are not present in this RNNParameterModel, raise an exception.
        superfluous_parameter_names = set(parameter_string_dict.keys()) - set(self.names)
        if len(superfluous_parameter_names) > 0:
            raise Exception('Superfluous parameters: {0}'.format(superfluous_parameter_names))

        print 'parsed parameter string dict = {0}'.format(retval)

        return retval

rnn_parameter_model = \
    RNNParameterModel({ \
        'data-class-name':{ \
            'description'          : 'an identifier for this class of data (e.g. c-code, english-text, etc)', \
            'value-type'           : str, \
            'validator'            : (lambda v : set(v) <= set('abcdefghijklmnopqrstuvwxyz0123456789-')), \
            'default-value'        : None, \
            'value-from-string'    : (lambda s : s), \
            'validity-description' : 'must be a string containing only hypens and lowercase alphanumeric characters', \
        }, \
        'training-data-filename':{ \
            'description'          : 'the name of a file containing the sequential training data on which to train the RNN', \
            'value-type'           : str, \
            'validator'            : (lambda v : True), \
            'default-value'        : None, \
            'value-from-string'    : (lambda s : s), \
            'validity-description' : 'must be a string specifying a valid filename', \
        }, \
        'hidden-layer-count':{ \
            'description'          : 'the number of hidden layers in the RNN', \
            'value-type'           : int, \
            'validator'            : (lambda v : v > 0), \
            'default-value'        : 2, \
            'value-from-string'    : (lambda s : int(s)), \
            'validity-description' : 'must be a positive integer', \
        }, \
        'hidden-layer-size':{ \
            'description'          : 'the size of each hidden layer in the RNN', \
            'value-type'           : int, \
            'validator'            : (lambda v : v > 0), \
            'default-value'        : 512, \
            'value-from-string'    : (lambda s : int(s)), \
            'validity-description' : 'must be a positive integer', \
        }, \
        'dropout':{ \
            'description'          : 'the dropout factor for training the RNN', \
            'value-type'           : float, \
            'validator'            : (lambda v : 0.0 <= v <= 1.0), \
            'default-value'        : 0.2, \
            'value-from-string'    : (lambda s : float(s)), \
            'validity-description' : 'must be a numeric value between 0.0 and 1.0', \
        }, \
        'epoch-count':{ \
            'description'          : 'the number of epochs this RNN has been trained for (or in the case of performing training, how many it should be trained for)', \
            'value-type'           : int, \
            'validator'            : (lambda v : v > 0), \
            'default-value'        : 1, \
            'value-from-string'    : (lambda s : int(s)), \
            'validity-description' : 'must be a positive integer', \
        }, \
        'training-sequence-count':{ \
            'description'          : 'the number of training input sequences to generate from the training data', \
            'value-type'           : int, \
            'validator'            : (lambda v : v > 0), \
            'default-value'        : None, \
            'value-from-string'    : (lambda s : int(s)), \
            'validity-description' : 'must be a positive integer', \
        }, \
        'training-sequence-size':{ \
            'description'          : 'the length of the training input sequences', \
            'value-type'           : int, \
            'validator'            : (lambda v : v > 0), \
            'default-value'        : None, \
            'value-from-string'    : (lambda s : int(s)), \
            'validity-description' : 'must be a positive integer', \
        }, \
    })

rnn_filename_prefix          = 'ramble.'
rnn_filename_suffix          = '.rnn'
rnn_metadata_filename_suffix = '.metadata.pickle'
rnn_report_filename_suffix = '.report.txt'

def parse_rnn_filename (filename):
    if not (filename[:len(rnn_filename_prefix)] == rnn_filename_prefix and filename[-len(rnn_filename_suffix):] == rnn_filename_suffix):
        return None

    # Retrieve the string between the prefix and suffix -- this is the part that specifies the parameters.
    filename_parameter_string = filename[len(rnn_filename_prefix):-len(rnn_filename_suffix)]

    # sys.stdout.write('    filename = {0}; filename_parameter_string = {1}\n'.format(filename, filename_parameter_string))
    parameter_string_dict = {}
    for param in filename_parameter_string.split(','):
        name_value_split = param.split(':')
        if len(name_value_split) != 2:
            sys.stdout.write('    Ignoring malformed .rnn filename "{0}".\n'.format(filename))
        else:
            name = name_value_split[0]
            value_string = name_value_split[1]
            parameter_string_dict[name] = value_string

    return rnn_parameter_model.parse_parameter_string_dict(parameter_string_dict)

def generate_rnn_filenames (rnn_parameters):
    rnn_weights_filename = rnn_filename_prefix + ','.join('{0}:{1}'.format(name,rnn_parameters[name]) for name in sorted(rnn_parameters.keys())) + rnn_filename_suffix
    rnn_metadata_filename = rnn_filename_prefix + ','.join('{0}:{1}'.format(name,rnn_parameters[name]) for name in sorted(rnn_parameters.keys()) if name != 'epoch-count') + rnn_metadata_filename_suffix
    rnn_report_filename = rnn_filename_prefix + ','.join('{0}:{1}'.format(name,rnn_parameters[name]) for name in sorted(rnn_parameters.keys()) if name != 'epoch-count') + rnn_report_filename_suffix
    return rnn_weights_filename,rnn_metadata_filename,rnn_report_filename

def generate_text_report_of_metadata (metadata):
    retval = 'RNN parameters:\n'
    for k,v in metadata['rnn-parameters'].iteritems():
        retval += '    {0} : {1}\n'.format(k,v)
    retval += '\n'

    training_analysis_history = metadata['training-analysis-history']

    if len(training_analysis_history) == 0:
        retval += 'No training has been done yet.\n'
    else:
        max_epoch_index = max(training_analysis_history.keys())
        retval += 'Most recent epoch has index {0}, training accuracy {1}, and average 0-sequence length {2}.\n'.format(max_epoch_index, training_analysis_history[max_epoch_index]['accuracy'], training_analysis_history[max_epoch_index]['average-sequence-lengths'][0])
        retval += '\n'

        retval += 'Summary of training accuracy with respect to epoch index:\n'
        for epoch_index,analysis in training_analysis_history.iteritems():
            retval += '    epoch {0} : accuracy {1}, average length of 0-sequences is {2}\n'.format(epoch_index, analysis['accuracy'], analysis['average-sequence-lengths'][0])
        retval += '\n'

        retval += 'Full report of analysis for each training epoch (in reverse order so the latest epoch is first):\n'
        retval += '\n'

        for epoch_index,analysis in reversed(list(training_analysis_history.iteritems())):
            retval += 'Epoch {0} analysis:\n'.format(epoch_index)

            retval += '    Rank frequencies:\n'
            for rank,frequency in enumerate(analysis['rank-frequencies']):
                retval += '        Rank {0}: {1}\n'.format(rank, frequency)
            retval += '    Sequence length histograms:\n'
            for symbol_class,sequence_length_histogram in enumerate(analysis['sequence-length-histograms']):
                retval += '        Symbol class {0} has average sequence length {1}.  Histogram:\n'.format(symbol_class, analysis['average-sequence-lengths'][symbol_class])
                for sequence_length in sorted(sequence_length_histogram.keys()):
                    retval += '            Length {0}: {1} occurrences\n'.format(sequence_length, sequence_length_histogram[sequence_length])

            retval += '\n'

    return retval

def generate_rnn_parameters_key (rnn_parameters):
    return tuple((name,rnn_parameters[name]) for name in sorted(rnn_parameters.keys()))

def keys_match_on_all_but_epoch_count (lhs_key, rhs_key):
    if len(lhs_key) != len(rhs_key):
        return False

    for ((lhs_name,lhs_value),(rhs_name,rhs_value)) in itertools.izip(lhs_key,rhs_key):
        if lhs_name != rhs_name:
            return False
        elif lhs_name != 'epoch-count':
            if lhs_value != rhs_value:
                return False

    # Everything matched except possibly the values for epoch-count.
    return True

def epoch_count_for_key (key):
    for name,value in key:
        if name == 'epoch-count':
            return value
    return None

def generate_alphabet (corpus):
  return sorted(list(set(corpus)))

def generate_alphabetical_symbol_index (alphabet):
    alphabetical_symbol_index = [0 for _ in range(256)]
    for i,symbol in enumerate(alphabet):
        alphabetical_symbol_index[ord(symbol)] = i
    for symbol in alphabet:
        assert symbol == alphabet[alphabetical_symbol_index[ord(symbol)]]
    assert all(0 <= symbol_index < len(alphabet) for symbol_index in alphabetical_symbol_index)
    return alphabetical_symbol_index

def load_training_data_and_alphabet (training_data_filename):
    # Read in the training data file and generate the alphabet of recognizable chars.
    with open(training_data_filename, 'r') as f:
        sys.stdout.write('Attempting to read training data file "{0}"...\n'.format(training_data_filename))
        start_time = time.time()
        training_data = f.read()
        sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    # TODO: Figure out if alphabet [size] should be an explicit parameter of the RNN.
    alphabet = generate_alphabet(training_data)
    alphabetical_symbol_index = generate_alphabetical_symbol_index(alphabet)
    sys.stdout.write('Alphabet has {0} symbols and is "{1}".\n'.format(len(alphabet), alphabet))
    sys.stdout.write('Alphabetical symbol index: {0}.\n'.format(alphabetical_symbol_index))

    return training_data,alphabet,alphabetical_symbol_index

def generate_and_load_model (rnn_parameters, alphabet, alphabetical_symbol_index):
    """
    Returns (model,trained_data_filename,epoch_start_index), where model is the compiled Keras-based
    RNN model, and trained_data_filename is the filename of the previously stored weights that were
    loaded into this model, or None if no appropriate file was found, and epoch_start_index is one
    greater than the epoch corresponding to trained_data_filename.
    """
    # # TODO: consider requiring that the training data filename specify the data class name.
    hidden_layer_count      = rnn_parameters['hidden-layer-count']
    hidden_layer_size       = rnn_parameters['hidden-layer-size']
    dropout                 = rnn_parameters['dropout']

    # Attempt to read pre-existing weight data (i.e. previously trained data upon which we can improve).
    sys.stdout.write('Looking for pre-existing trained RNN data...\n')
    start_time = time.time()
    filenames = os.listdir('.')
    # TODO: put this in a 'rnn file directory reading' function
    trained_rnn_directory = {} # Indexed by data-class-name
    for filename in filenames:
        try:
            filename_rnn_parameters = parse_rnn_filename(filename)
            if filename_rnn_parameters == None:
                continue
            # sys.stdout.write('    filename = {0}\n'.format(filename))
            # sys.stdout.write('    rnn parameters = {0}\n'.format(filename_rnn_parameters))
            rnn_weights_filename,rnn_metadata_filename,_ = generate_rnn_filenames(filename_rnn_parameters)
            # sys.stdout.write('    generate_rnn_filename(rnn_parameters) = {0}\n'.format(rnn_weights_filename))
            if rnn_weights_filename != filename:
                sys.stdout.write('    Warning: The rnn_parameters for .rnn filename\n        {0}\n    are not in canonical representation; it should be\n        {1}\n'.format(filename, rnn_weights_filename))
            filename_rnn_parameters_key = generate_rnn_parameters_key(filename_rnn_parameters)
            trained_rnn_directory[filename_rnn_parameters_key] = rnn_weights_filename,rnn_metadata_filename
        except Exception as e:
            sys.stdout.write('    Ignoring malformed .rnn filename "{0}" ({1}).\n'.format(filename, str(e)))
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    # Determine if there is pre-existing rnn data that can be loaded, instead of starting from scratch.
    rnn_parameters_key = generate_rnn_parameters_key(rnn_parameters)
    sorted_matching_training_keys = sorted([key for key in trained_rnn_directory.keys() if keys_match_on_all_but_epoch_count(key, rnn_parameters_key)], reverse=True, key=epoch_count_for_key)
    # sys.stdout.write('sorted_matching_training_keys = {0}\n'.format(sorted_matching_training_keys))
    if len(sorted_matching_training_keys) > 0:
        best_trained_rnn_parameters_key = sorted_matching_training_keys[0]
        trained_data_filename,metadata_filename = trained_rnn_directory[best_trained_rnn_parameters_key]
        # Delete all obsolete files.
        for obsolete_trained_rnn_parameters_key in sorted_matching_training_keys[1:]:
            os.remove(trained_rnn_directory[obsolete_trained_rnn_parameters_key][0])
            os.remove(trained_rnn_directory[obsolete_trained_rnn_parameters_key][1])
        epoch_start_index = epoch_count_for_key(best_trained_rnn_parameters_key) + 1 # Start one after the last one.
        sys.stdout.write('Found existing RNN file "{0}" from which to resume training starting at epoch {1}.\n'.format(trained_data_filename, epoch_start_index))
    else:
        trained_data_filename = None
        metadata_filename = None
        epoch_start_index = 1 # 0 is not a valid 'epoch-count' RNN parameter, so start at 1.
        sys.stdout.write('No pre-existing relevant trained RNN data.\n')

    # Construct the RNN and compile it (be verbose in everything, and give timing info).
    sys.stdout.write('Building RNN with hidden layer count = {0}, hidden layer size = {1}, and dropout = {2}...\n'.format(hidden_layer_count, hidden_layer_size, dropout))
    start_time = time.time()
    model = Sequential()
    model.add(LSTM(len(alphabet), hidden_layer_size, return_sequences=True))
    model.add(Dropout(dropout))
    for _ in range(hidden_layer_count-2):
        model.add(LSTM(hidden_layer_size, hidden_layer_size, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer_size, hidden_layer_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_size, len(alphabet)))
    model.add(Activation('softmax'))
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    sys.stdout.write('Compiling model...\n')
    start_time = time.time()
    # TODO: add loss and optimizer as parameters.
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

    # If there was a rnn file to load, do that now.
    if trained_data_filename != None:
        model.load_weights(trained_data_filename)

    return model,trained_data_filename,metadata_filename,epoch_start_index

# TEMP
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

def train (**kwargs):
    sys.stdout.write('train({0})\n'.format(kwargs))
    global_start_time = time.time()

    try:
        # Seed the random number generator so that everything here is reproducible.
        np.random.seed(666)

        # # TODO: consider requiring that the training data filename specify the data class name.
        rnn_parameters          = rnn_parameter_model.parse_parameter_string_dict(kwargs)
        training_data_filename  = rnn_parameters['training-data-filename']
        epoch_count             = rnn_parameters['epoch-count']
        training_sequence_count = rnn_parameters['training-sequence-count']
        training_sequence_size  = rnn_parameters['training-sequence-size']

        training_data,alphabet,alphabetical_symbol_index = load_training_data_and_alphabet(training_data_filename)

        model,trained_data_filename,metadata_filename,epoch_start_index = generate_and_load_model(rnn_parameters, alphabet, alphabetical_symbol_index)

        # Generate the training data.  Using random sequence start indices is probably not the best way to go.
        # TODO: Figure out a better way.
        sys.stdout.write('Generating {0} training sequences of length {1}...\n'.format(training_sequence_count, training_sequence_size))
        start_time = time.time()
        sequence_start_indices = np.random.random_integers(0, len(training_data)-training_sequence_size-2, size=training_sequence_count)
        # Use a one-hot encoding of each symbol -- this is the characteristic probability distribution for that symbol.
        X = np.zeros((training_sequence_count, training_sequence_size, len(alphabet)), dtype=np.bool)
        y = np.zeros((training_sequence_count, len(alphabet)), dtype=np.bool)
        # sys.stdout.write('sequences:\n')
        for sequence_index,sequence_start_index in enumerate(sequence_start_indices):
            # sys.stdout.write('    sequence {0} with start index {1}: "{2}" |-> "{3}"\n'.format(sequence_index, sequence_start_index, training_data[sequence_start_index:sequence_start_index+training_sequence_size], training_data[sequence_start_index+training_sequence_size]))
            for timestep_index in range(training_sequence_size):
                X[sequence_index, timestep_index, alphabetical_symbol_index[ord(training_data[sequence_start_index+timestep_index])]] = 1
            y[sequence_index, alphabetical_symbol_index[ord(training_data[sequence_start_index+training_sequence_size])]] = 1
        sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

        # Read the metadata file
        try:
            with open(metadata_filename, 'r') as f:
                metadata = pickle.load(f)
        except Exception as e:
            sys.stdout.write('No metadata file could be loaded (error was "{0}") -- creating new metadata file with empty training analysis history.\n'.format(str(e)))
            metadata = { \
                'rnn-parameters'           :rnn_parameters, \
                'alphabet'                 :alphabet, \
                'training-analysis-history':{}, \
            }

        # Train for the specified number of epochs.
        previous_rnn_filename = trained_data_filename
        for epoch_index in range(epoch_start_index, epoch_start_index+epoch_count):
            sys.stdout.write('Training; epoch {0}...\n'.format(epoch_index))
            start_time = time.time()
            epoch_start_time = start_time
            rnn_parameters['epoch-count'] = epoch_index
            rnn_filename,metadata_filename,report_filename = generate_rnn_filenames(rnn_parameters)
            fit_retval = model.fit(X, y, batch_size=128, nb_epoch=1, show_accuracy=True, validation_split=0.1)
            epoch_end_time = time.time()
            sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

            # Write the state of the RNN to file.
            sys.stdout.write('Writing state of RNN to file...\n'.format(epoch_index))
            start_time = time.time()
            model.save_weights(rnn_filename, overwrite=True)
            sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

            # Delete the old RNN file, if it exists
            if previous_rnn_filename != None:
                sys.stdout.write('Deleting obsolete RNN file...\n')
                start_time = time.time()
                os.remove(previous_rnn_filename)
                sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))
            previous_rnn_filename = rnn_filename

            # Sample from the RNN a bit
            samples = {}
            sampling_sequence_start_index = 0
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                seed = training_data[sampling_sequence_start_index:sampling_sequence_start_index+training_sequence_size]
                sys.stdout.write('----- Diversity: {0}, seed: "{1}"\n'.format(diversity, seed))

                generated = ''
                generated += seed
                # print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for iteration in range(400):
                    x = np.zeros((1, training_sequence_size, len(alphabet)))
                    for timestep_index,symbol in enumerate(seed):
                        x[0, timestep_index, alphabetical_symbol_index[ord(symbol)]] = 1.0

                    preds = model.predict(x, verbose=0)[0]
                    next_symbol_index = sample(preds, diversity)
                    next_symbol = alphabet[next_symbol_index]

                    generated += next_symbol
                    seed = seed[1:] + next_symbol

                    sys.stdout.write(next_symbol)
                    sys.stdout.flush()

                samples[diversity] = generated
                sys.stdout.write('\n')

            # Generate the rank sequence for training_data, and compute the statistics about rank
            # frequencies and sequence lengths.
            sys.stdout.write('Generating rank sequence...\n')
            start_time = time.time()
            # TODO: generate histogram of rank likelihoods.  this would give a measure of how surprising
            # the input stream is with respect to the RNN.
            rank_sequence_generation_start_time = time.time()
            rank_sequence = generate_rank_sequence_2(model, training_data[:10000], training_sequence_size, alphabet, alphabetical_symbol_index)
            rank_sequence_generation_end_time = time.time()
            rank_frequencies = compute_frequencies(rank_sequence, len(alphabet))
            symbol_class_count = 2
            symbol_class_sequence = [min(symbol_class_count-1,rank) for rank in rank_sequence]
            sequence_length_histograms = compute_sequence_length_histograms(symbol_class_sequence, symbol_class_count)
            average_sequence_lengths = compute_average_sequence_lengths(sequence_length_histograms)
            sys.stdout.write('    ...finished in {0} seconds.\n'.format(time.time() - start_time))

            sys.stdout.write('rank frequencies:\n')
            for rank,frequency in enumerate(rank_frequencies):
                sys.stdout.write('    rank {0}: {1}\n'.format(rank, frequency))
            sys.stdout.write('sequence length histograms:\n')
            for symbol_class,sequence_length_histogram in enumerate(sequence_length_histograms):
                sys.stdout.write('    symbol class {0} has average sequence length {1}.  histogram:\n'.format(symbol_class, average_sequence_lengths[symbol_class]))
                for sequence_length,count in sequence_length_histogram.iteritems():
                    sys.stdout.write('        length {0}: {1} occurrences\n'.format(sequence_length, count))

            # Store the training analysis in the metadata.
            analysis_for_this_epoch = { \
                'training-epoch-duration':epoch_end_time-epoch_start_time, \
                'accuracy':fit_retval.history['acc'][0], \
                'loss':fit_retval.history['loss'][0], \
                'samples':samples, \
                'rank-frequencies':rank_frequencies, \
                'average-sequence-lengths':average_sequence_lengths, \
                'sequence-length-histograms':sequence_length_histograms, \
                'rank-sequence-generation-duration':rank_sequence_generation_end_time-rank_sequence_generation_start_time, \
            }
            assert epoch_index not in metadata['training-analysis-history']
            metadata['training-analysis-history'][epoch_index] = analysis_for_this_epoch

            # Write out the metadata
            try:
                with open(metadata_filename, 'w') as f:
                    pickle.dump(metadata, f)
            except Exception as e:
                sys.stdout.write('Failure while writing metadata file "{0}".  Error was "{1}".\n'.format(metadata_filename, str(e)))

            report = generate_text_report_of_metadata(metadata)
            try:
                with open(report_filename, 'w') as f:
                    f.write(report)
            except Exception as e:
                sys.stdout.write('Failure while writing report file "{0}".  Error was "{1}".\n'.format(report_filename, str(e)))

            # TODO: should probably write a metadata file of non-parameters for the human's benefit
            # e.g. loss value, accuracy, etc.

    finally:
        # This is inside a try/finally clause so that KeyboardInterrupt, while it quits the program,
        # still allows the total time spent to be printed.
        sys.stdout.write('\nTotal time in this program (including possible pauses) was {0} seconds.\n'.format(time.time() - global_start_time))

    return 0

def main (argv):
    try:
        opts,args = \
            getopt.getopt( \
                argv[1:], \
                't', \
                [ \
                    'data-class-name=', \
                    'epoch-count=', \
                    'hidden-layer-count=', \
                    'hidden-layer-size=', \
                    'train', \
                    'training-data-filename=', \
                    'training-sequence-count=', \
                    'training-sequence-size=', \
                ] \
            )

        if len(opts) == 0:
            sys.stdout.write(ramble_usage_string(argv))
            return 0

        sys.stdout.write('opts = {0}\n'.format(opts))
        for opt in opts:
            sys.stdout.write('opt = {0}\n'.format(opt))
            if opt[0] == '-t' or opt[0] == '--train':
                action = train
                action_args = {}
            elif opt[0] == '--data-class-name':
                action_args['data-class-name'] = opt[1]
            elif opt[0] == '--epoch-count':
                action_args['epoch-count'] = opt[1]
            elif opt[0] == '--hidden-layer-count':
                action_args['hidden-layer-count'] = opt[1]
            elif opt[0] == '--hidden-layer-size':
                action_args['hidden-layer-size'] = opt[1]
            elif opt[0] == '--training-data-filename':
                action_args['training-data-filename'] = opt[1]
            elif opt[0] == '--training-sequence-count':
                action_args['training-sequence-count'] = opt[1]
            elif opt[0] == '--training-sequence-size':
                action_args['training-sequence-size'] = opt[1]
            else:
                assert False, 'this should never happen'

        return action(**action_args)
    except getopt.GetoptError as e:
        sys.stdout.write('error: {0}\n'.format(str(e)))
        sys.stdout.write(ramble_usage_string(argv))
        return -1

if __name__ == '__main__':
    sys.exit(main(sys.argv))

