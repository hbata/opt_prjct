from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt


def djmodel(Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """
    # Define the input of the model with a shape
    X = Input(shape=(Tx, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    # Step 1: Create empty list to append the outputs while you iterate
    outputs = []
    for t in range(Tx):
        # Step 2.A: select the "t"th time step vector from X.
        x = Lambda(lambda x: X[:, t, :])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)

    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model()
    densor -- the trained "densor" from model()
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model
    """

    # Define the input of your model with a shape
    x0 = Input(shape=(1, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    # Step 1: Create an empty list of "outputs" to later store your predicted values
    outputs = []
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Step 2.A: Perform one step of LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)
        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78)
        outputs.append(out)
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step.
        #           We have provided
        #           the line of code you need to do this.
        x = Lambda(one_hot)(out)
    # Step 3: Create model instance with the correct "inputs" and "outputs"
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return inference_model


def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cell

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    # Step 1: Use the inference model to predict an output sequence given x_initializer,
    # a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=78)
    return results, indices


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()


# LSTM with 64 dimensional hidden states
n_a = 64
# Load music values
X, Y, n_values, indices_values = load_music_utils()
# LSTM global layers:
reshapor = Reshape((1, 78))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(n_values, activation='softmax')

print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)
# define the model.
model = djmodel(Tx=30, n_a=64, n_values=78)
# Compile the model to be trained
# Using Adam optimizer and a categorical cross-entropy loss.
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# initialize a0 and c0 for the LSTM's initial state to be zero.
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
# Fit the model:
# Turn Y to a list before the training, since the cost function expects Y to be provided in this format
# (one list item per time-step), so:
# list(Y) is a list with 30 items, where each of the list items is of shape (60,78).
# Train for 100 epochs.(epoch: 1 pass through the training set)
history = model.fit([X, a0, c0], list(Y), epochs=100)
plot_history(history)
# plot training:
print(history.history.keys())
#  "Accuracy"
# plt.figure('training accuracy')
# plt.plot(history.history['dense_1_acc_30'])
# # plt.plot(history.history['val_acc'])
# plt.title('model accuracy (Adam)')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# # "Loss"
# plt.figure('training loss')
# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('model loss (Adam)')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# Define your inference model.
# This model is hard coded to generate 50 values.
inference_model = music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50)
# Create the zero-valued vectors you will use to initialize x and the LSTM state variables a and c
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))
# RNN generates a sequence of values. The code generates music
# by first calling the predict_and_sample() function.
# Most computational music algorithms use some post-processing because it is difficult to generate music
#  that sounds good without such post-processing. The post-processing does things such as clean up
# the generated audio by making sure the same sound is not repeated too many times, that two successive notes
#  are not too far from each other in pitch, and so on.
#  A lot the music generation literature has also focused on hand-crafting post-processors,
# and a lot of the output quality depends on the quality of the post-processing
# and not just the quality of the RNN.
# generate music and record it into the out_stream.
out_stream = generate_music(inference_model)
