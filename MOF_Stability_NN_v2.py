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

    # Move Data from Excel File (x already normalized and MOFs randomized in Excel) into Array
    for j in range(1, num_MOFs + 1):
        for i in range(4, dataset.ncols):
            data[j - 1, i - 4] = dataset.cell_value(j, i)

    for i in range(4, dataset.ncols - 1):
        cat.append(dataset.cell_value(0, i))

    # Separate Training and Testing Data Sets
    x_tr = data[0:23, 0:8]
    y_tr = data[0:23, 8]
    x_tst = data[24:num_MOFs - 1, 0:8]
    y_tst = data[24:num_MOFs - 1, 8]

    return (x_tr, y_tr), (x_tst, y_tst), cat


def set_categories(x_tr, y_tr, x_tst, y_tst, categories):

    num_categories = list(range(0, len(categories)))
    poss_combos = combinations(num_categories, 3)
    results = []
    print("X Variables \t \t Training MSE \t Testing MSE")

    for i in poss_combos:
        x_training = x_tr[:, [i[0], i[1], i[2]]]
        x_testing = x_tst[:, [i[0], i[1], i[2]]]
        name = str(categories[i[0]] + " + " + categories[i[1]] + " + " + categories[i[2]])
        # Build network
        model, history = build_network(x_training, y_tr, x_testing, y_tst)
        # Observe the performance of the network
        trial_data = plot_performance(model, history, x_testing, y_tst, name, results)

    data_file = open("4_layer_relu_0.2_10_8_6_6.txt", 'w')
    for line in trial_data:
        data_file.write(str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\n")

    data_file.close()


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
    model.add(Dropout(0.15))
    # Add second layer. D2on't have to specify shape any more.
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    # Add final layer, reducing # of nodes to 1 corresponding to estimated thermal decomposition temperature/100
    model.add(Dense(1))
    model.add(Activation('linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=3, epochs=100, verbose=0, validation_data=(x_test, y_test))

    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    #print('Saved trained model at %s ' % model_path)

    # Return the model
    return model, history


def plot_performance(model, history, x_test, y_test, output_name, results):
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

    #Calculate Loss and ccuracy on the Training Data
    tr_loss = history.history['loss'][len(history.epoch) - 1]

    # Calculate loss and accuracy on the testing data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    results.append([output_name, tr_loss, loss])
    print(output_name + "\t \t" + str(round(tr_loss, 3)) + "\t \t \t" + str(round(loss, 3)))
    # Save figure as .png file
    #fig = plt.gcf()
    #fig.savefig(output_name)

    return results


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load data into appropriate training and testing variables
    (x_train, y_train), (x_test, y_test), categories = load_data("ThermalStability.xlsx")

    #n_classes = 6
    #y_train = np_utils.to_categorical(y_train, n_classes)
    #y_test = np_utils.to_categorical(y_test, n_classes)

    set_categories(x_train, y_train, x_test, y_test, categories)
