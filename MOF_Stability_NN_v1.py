'''
Dataset contains information about metals centers, organic linkers, and structures of various MOFs. The temperature at
which decomposition is observed in thermo-gravimetric analysis (TGA) was used to determine the thermal stability of the
MOF and the stability of the MOF with respect to the medium it was stored in was also considered [0 is air sensitive; 1
is air stable, but degrades in liquid water; 2 is stable in room temperature water, but degrades in mildly acidic and/or
basic water and/or boiling water; 3 is stable in all the above conditions].
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import xlrd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from itertools import combinations

def load_data(fptr):
    '''

     **Parameters**

        fptr: *str*
            name of file containing data

    **Returns**
        x_tr: *numpy.ndarray*
            array containing domain of training data set
        y_tr: *numpy.ndarray*
            array containing range of training data set

        x_tst: *numpy.ndarray*
            array containing domain of testing data set
        y_tst: *numpy.ndarray*
            array containing range of testing data set

    '''

    book = xlrd.open_workbook(fptr)

    dataset = book.sheet_by_index(0)
    book.release_resources()
    num_MOFs = dataset.nrows - 1


    # Initialize Matrices
    data = np.empty((num_MOFs, 9))
    cat = []

    for j in range(1, num_MOFs + 1):
            for i in range(4, dataset.ncols):
                data[j - 1, i - 4] = dataset.cell_value(j, i)

    for i in range(4, dataset.ncols - 1):
        cat.append(dataset.cell_value(0, i))

    # Randomize arrays
    np.random.shuffle(data)

    # Separate Training and Testing Data Sets
    x_tr = data[0:23, 0:7]
    y_tr = data[0:23, 8]
    x_tst = data[24:num_MOFs - 1, 0:7]
    y_tst = data[24:num_MOFs - 1, 8]

    return (x_tr, y_tr), (x_tst, y_tst), cat


def set_categories(x_tr, y_tr, x_tst, y_tst, categories):

    num_categories = list(range(0, len(categories) - 1))
    poss_combos = combinations(num_categories, 3)

    for i in poss_combos:
        x_train = x_tr[:, [i[0], i[1], i[2]]]
        x_test = x_tst[:, [i[0], i[1], i[2]]]
        name = str("1." + categories[i[0]] + "  2." + categories[i[1]] + "  3." + categories[i[2]] + ".jpg")
        # Build network
        model, history = build_network(x_train, y_tr, x_test, y_tst)
        # Observe the performance of the network
        plot_performance(model, history, x_test, y_test, name)


def build_network(x_train, y_train, x_test, y_test,
                  model_name='keras_MOF_NN.h5'):
    '''
    This function holds the neural network model as built by you, the user.
    After building the model, the model is then trained and saved to a .h5
    file.

    **Parameters**

        x_train: *numpy.ndarray*
            The set of training data, which should be an array of size
            (60000, 784).
        y_train: *numpy.ndarray*
            The set of labels for the corresponding training data, which
            should be an array of size (60000, 10).
        x_test: *numpy.ndarray*
            The set of testing data, which should be an array of size
            (10000, 784).
        y_test: *numpy.ndarray*
            The set of labels for the corresponding testing data, which should
            be an array of size (10000, 10).
        model_name: *str, optional*
            The filename of the model to be saved.

    **Returns**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.

    '''
    # Build the model (Layer by Layer)
    model = Sequential()
    # Add first layer with input as vector with an index for each pixel
    model.add(Dense(10, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # Add second layer. Don't have to specify shape any more.
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # Add final layer, reducing # of nodes to 10 corresponding to digits 0-9
    model.add(Dense(6))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=24, epochs=100, verbose=0, validation_data=(x_test, y_test))

    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Return the model
    return model, history


def plot_performance(model, history, x_test, y_test, output_name):
    '''
    Retrieve accuracies from the history object and save them to a figure.

    **Parameters**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.
        output_name: *str, optional*
            The filename of the output image.

    **Returns**

        None
    '''
    # Plot accuracy
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    # Calculate loss and accuracy on the testing data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    # Save figure as .png file
    fig = plt.gcf()
    fig.savefig(output_name)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load data into appropriate training and testing variables
    (x_train, y_train), (x_test, y_test), categories = load_data("ThermalStability.xlsx")

    n_classes = 6
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    set_categories(x_train, y_train, x_test, y_test, categories)
