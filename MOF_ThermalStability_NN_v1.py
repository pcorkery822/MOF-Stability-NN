'''
Dataset contains information about metals centers, organic linkers, and structures of various MOFs. The temperature at
which decomposition is observed in thermo-gravimetric analysis (TGA) was used to determine the thermal stability of the
MOF.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import xlrd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from itertools import combinations
from operator import itemgetter


def load_data(fptr):
    '''
    Loads data from an Excel file containing possible parameters (Linker functional group classification, linker pKa,
    Metal d electron count, metal electronegativity, metal charge in MOF complex, linker molecular weight, number of
    atoms coordinated to metal atoms within a single linker molecule, and Langmuir surface area) to use as inputs to NN,
    MOF names to use as labels, and thermal decomposition temperatures for MOFs used as outputs.

     **Parameters**

        fptr: *str*
            name of file containing data

    **Returns**
        x_train: *numpy.ndarray*
            array containing domain of training data set
        y_train: *numpy.ndarray*
            array containing range of training data set

        x_test: *numpy.ndarray*
            array containing domain of testing data set
        y_test: *numpy.ndarray*
            array containing range of testing data set

        param: *list*
            list of parameters
        mof: *list*
            list of names for the MOFs in the dataset

    '''

    book = xlrd.open_workbook(fptr)

    dataset = book.sheet_by_index(0)
    book.release_resources()
    num_MOFs = dataset.nrows - 1

    # Initialize lists and arrays
    data = np.empty((num_MOFs, 9))
    params = []
    mofs = []

    # Move Data from Excel File (x already normalized and MOFs randomized in Excel) into Array
    for j in range(1, num_MOFs + 1):
        mofs.append(dataset.cell_value(j, 0))
        for i in range(4, dataset.ncols):
            data[j - 1, i - 4] = dataset.cell_value(j, i)

    for i in range(4, dataset.ncols - 1):
        params.append(dataset.cell_value(0, i))

    # Separate Training and Testing Data Sets
    x_train = data[0:24, 0:8]
    y_train = data[0:24, 8]
    x_test = data[24:num_MOFs, 0:8]
    y_test = data[24:num_MOFs, 8]

    return (x_train, y_train), (x_test, y_test), params, mofs


def evaluate_xvars(x_train, y_train, x_test, y_test, params, filename='ThermalStability_NN.txt'):
    '''
    This function generates NNs for all possible combinations of three parameters at a time for all eight parameters
    (8 choose 3 = 56). It then compares the mean squared error for each combination to determine which three parameters
    should be used to most accurately model the system.

     **Parameters**
        x_train: *numpy.ndarray*
            array containing domain of training data set
        y_train: *numpy.ndarray*
            array containing range of training data set

        x_test: *numpy.ndarray*
            array containing domain of testing data set
        y_test: *numpy.ndarray*
            array containing range of testing data set

        params: *numpy.ndarray*
            1D array of possible linker, metal, and MOF characteristics from dataset

    **Returns**
        x_train_opt: *numpy.ndarray*
            array containing training x data of the three parameters that contributed to the lowest mean squared error
            when used to generate the model. Should be a 2D array of 24 rows and three columns.

        x_test_opt
            array containing testing x data of the three parameters that contributed to the lowest mean squared error
            when used to generate the model. Should be a 2D array of 14 rows and three columns.

    '''

    num_params = list(range(0, len(params)))
    poss_combos = combinations(num_params, 3)
    results = []
    columns_list = []
    print("X Variables \t \t Training MSE \t Testing MSE")

    for i in poss_combos:
        x_training = x_train[:, [i[0], i[1], i[2]]]
        x_testing = x_test[:, [i[0], i[1], i[2]]]
        # List of Linker, Metal, or MOF properties evaluated in current iteration
        x_vars = [params[i[0]], params[i[1]], params[i[2]]]
        columns_list.append([i[0], i[1], i[2]])
        # Build network
        model, history = build_network(x_training, y_train, x_testing, y_test)
        # Observe the performance of the network
        trial_data = evaluate_performance(model, history, x_testing, y_test, x_vars, results)

    data_file = open(filename, 'w')
    for line in trial_data:
        data_file.write(str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\t" + str(line[3]) + "\t" + str(line[4]) + "\n")

    data_file.close()

    # Return index of best parameters to approximate the system. Best parameters judged by smallest mean squared error
    # of the model for the testing data.
    tr_mse_vals = [x[4] for x in trial_data]
    ind = tr_mse_vals.index(min(tr_mse_vals))
    print('Chosen parameters for optimized model :', trial_data[ind][0], trial_data[ind][1], trial_data[ind][2])

    # Set x values for finalized model
    x_train_opt = x_train[:, [columns_list[ind][0], columns_list[ind][1], columns_list[ind][2]]]
    x_test_opt = x_test[:, [columns_list[ind][0], columns_list[ind][1], columns_list[ind][2]]]

    return x_train_opt, x_test_opt


def build_network(x_train, y_train, x_test, y_test, nEpochs=1000):
    '''
    This function builds and trains the neural network.

    **Parameters**

        x_train: *numpy.ndarray*
            The domain of the training data. Should be a 2D array of 24 rows and three columns.
        y_train: *numpy.ndarray*
            The range of the training data. Should be a 1D array of length 24.
        x_test: *numpy.ndarray*
            The domain of the testing data. Should be a 2D array of 14 rows and three columns.
        y_test: *numpy.ndarray*
            The range of the testing data. Should be a 1D array of length 24.
        n_epoch: *int*
            Number of epochs for model. Default value of 1000.

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
    model.add(Dropout(0.5))
    # Add second layer. Don't have to specify shape any more.
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Add final layer, reducing # of nodes to 1 corresponding to estimated thermal decomposition temperature/100
    model.add(Dense(1))
    model.add(Activation('linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=3, epochs=nEpochs, verbose=0, validation_data=(x_test, y_test))

    # Return the model
    return model, history


def evaluate_performance(model, history, x_test, y_test, x_vars, results):
    '''
    This function determines the mean squared error for a given set of parameters and appends this data to a list
    containing the MSE associated with the other possible sets of parameters.

    **Parameters**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.
        x_test: *numpy.ndarray*
            The domain of the testing data. Should be a 2D array of 24 rows and three columns.
        y_test: *numpy.ndarray*
            The range of the testing data. Should be a 1D array of length 24.
        x_vars: *list*
            List containing set of parameters considered for current iteration of the model.
        results: *list*
            List containing the set of three columns of x values used for current iteration of model and the
            corresponding MSE.

    **Returns**

        results: *list*
            List containing the set of three columns of x values used for current iteration of model and the
            corresponding MSE.
    '''

    # Calculate mean squared error of model with the training Data
    train_mse = history.history['loss'][len(history.epoch) - 1]

    # Calculate mean squared error of the model with the testing data
    test_mse = model.evaluate(x_test, y_test, verbose=2)
    results.append([x_vars[0], x_vars[1], x_vars[2], train_mse, test_mse])
    print(x_vars[0] + "+" + x_vars[1] + "+" + x_vars[2] + "\t \t \t" + str(round(train_mse, 0)) + "\t \t \t"
          + str(round(test_mse, 0)))

    return results


def finalize_model(x_train, y_train, x_test, y_test, mofs, model_name='MOF_ThermalStability_120320_NN.h5',
                   output_name='MOF_ThermalStability_Plot.png', output_name2='MOF_ThermalStability_Loss.png'):
    '''
    This function builds and trains a neural network using the optimal combination of parameters. It then plots the
    predicted thermal decomposition temperature values as function of the true value and the mean squared error of the
    model for the training and testing data sets as a function of the epoch. The function also saves both plots and the
    model.

    **Parameters**

        x_train: *numpy.ndarray*
            The domain of the training data. Should be a 2D array of 24 rows and three columns.
        y_test: *numpy.ndarray*
            The range of the training data. Should be a 1D array of length 24.
        x_test: *numpy.ndarray*
            The domain of the testing data. Should be a 2D array of 14 rows and three columns.
        y_test: *numpy.ndarray*
            The range of the testing data. Should be a 1D array of length 24.
        mofs: *list*
            list of names for the MOFs in the dataset.
        model_name: *str*
            name of save file for model.
        output_name: *str*
            name of save file for plots of true and predicted thermal decomposition temperatures.
        output_name2: *str*
            name of save file for plot of loss as a function of epoch.

    **Returns**

    '''
    # Generate model for chosen parameters using 1000 epochs
    model, history = build_network(x_train, y_train, x_test, y_test, 5000)

    # Generate predicted values with the improved model
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Combine and sort data sets by true thermal decomposition temperature
    train = []
    test = []
    for i, j in enumerate(y_train):
        train.append([mofs[i], j, float(train_pred[i])])

    for i, j in enumerate(y_test):
        test.append([mofs[i + len(y_train) - 1], j, float(test_pred[i])])

    train_sort = sorted(train, key=itemgetter(1))
    test_sort = sorted(test, key=itemgetter(1))

    # Plot predicted values as a function of actual values
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot([z[1] for z in train_sort], 'r^')
    plt.plot([z[2] for z in train_sort], 'bo')
    x = list(range(len(y_train)))
    plt.ylim(0, 600)
    train_labels = [z[0] for z in train_sort]
    train_labels = [z.replace(' ', '\n') for z in train_labels]
    plt.xticks(x, train_labels, rotation=45, fontsize=6)
    plt.title('Training Data')
    plt.ylabel('Temperature (\N{DEGREE SIGN}Celsius)')
    plt.legend(['True', 'Predicted'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot([z[1] for z in test_sort], 'r^')
    plt.plot([z[2] for z in test_sort], 'bo')
    y = list(range(len(y_test)))
    test_labels = [z[0] for z in test_sort]
    test_labels = [z.replace(' ', '\n') for z in test_labels]
    plt.xticks(y, test_labels, rotation=45, fontsize=6)
    plt.ylim(0, 600)
    plt.title('Testing Data')
    plt.ylabel('Temperature (\N{DEGREE SIGN}Celsius)')
    plt.legend(['True', 'Predicted'], loc='lower right')
    plt.tight_layout(pad=0.3, w_pad=0, h_pad=0)
    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(output_name)

    # Plot Mean Squared Error vs Epoch
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Mean Squared Error')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(output_name2)

    # Calculate loss and accuracy on the testing data
    mse = model.evaluate(x_test, y_test, verbose=0)
    print("Test Loss:", mse)

    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load data into appropriate training and testing variables
    (x_train, y_train), (x_test, y_test), params, mofs = load_data("ThermalStability.xlsx")

    # Determine which three parameters are best for predicting thermal decomposition temperature
    x_train_opt, x_test_opt = evaluate_xvars(x_train, y_train, x_test, y_test, params)

    # Perform more extensive analysis of chosen parameters, save model, and produce plots of MSE and predicted values
    finalize_model(x_train_opt, y_train, x_test_opt, y_test, mofs)
