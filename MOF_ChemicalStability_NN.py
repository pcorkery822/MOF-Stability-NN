'''
Dataset contains information about metals centers, organic linkers, and structures of various MOFs. The stability of the
MOF with respect to the medium it was stored in is also contained within the dataset and was treated as the y variable
for the models here [0 is air sensitive; 1 is air stable, but degrades in liquid water; 2 is stable in room temperature
water, but degrades in mildly acidic and/or basic water and/or boiling water; 3 is stable in all the above conditions].
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
    Loads data from an Excel file containing possible parameters (Linker functional group classification, linker pKa,
    Metal d electron count, metal electronegativity, metal charge in MOF complex, linker molecular weight, number of
    atoms coordinated to metal atoms within a single linker molecule, and Langmuir surface area) to use as inputs to NN,
    MOF names to use as labels, and chemical stability classification for MOFs used as outputs.

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
    x_train = data[0:24, 0:8]
    y_train = data[0:24, 8]
    x_test = data[24:num_MOFs, 0:8]
    y_test = data[24:num_MOFs, 8]

    return (x_train, y_train), (x_test, y_test), cat


def evaluate_xvars(x_train, y_train, x_test, y_test, categories, filename='ThermalStability_NN.txt'):
    '''

     **Parameters**
        x_train: *numpy.ndarray*
            array containing domain of training data set
        y_train: *numpy.ndarray*
            array containing range of training data set

        x_test: *numpy.ndarray*
            array containing domain of testing data set
        y_test: *numpy.ndarray*
            array containing range of testing data set

        categories: *numpy.ndarray*
            1D array of possible linker, metal, and MOF characteristics from dataset

    **Returns**
        x_train_opt: *numpy.ndarray*
            array containing training x data of the three parameters that contributed to the lowest mean squared error
            when used to generate the model. Should be a 2D array of 24 rows and three columns.

        x_test_opt
            array containing testing x data of the three parameters that contributed to the lowest mean squared error
            when used to generate the model. Should be a 2D array of 14 rows and three columns.

    '''

    num_categories = list(range(0, len(categories)))
    poss_combos = combinations(num_categories, 3)
    results = []
    columns_list = []
    print("X Variables \t \t Training MSE \t Testing MSE")

    for i in poss_combos:
        x_training = x_train[:, [i[0], i[1], i[2]]]
        x_testing = x_test[:, [i[0], i[1], i[2]]]
        # List of Linker, Metal, or MOF properties evaluated in current iteration
        x_vars = [categories[i[0]], categories[i[1]], categories[i[2]]]
        columns_list.append([i[0], i[1], i[2]])
        # Build network
        model, history = build_network(x_training, y_train, x_testing, y_test, 1)
        # Observe the performance of the network
        trial_data = evaluate_performance(model, history, x_testing, y_test, x_vars, results)

    data_file = open(filename, 'w')
    for line in trial_data:
        data_file.write(str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\t" + str(line[3]) + "\t" + str(line[4]) + "\n")

    data_file.close()

    # Return index of best parameters to approximate the system. Best parameters judged by smallest mean squared error
    # of the model on the testing data.
    train_mse_vals = [x[3] for x in trial_data]
    ind = train_mse_vals.index(min(train_mse_vals))
    print('Chosen parameters for optimized model :', trial_data[ind][0], trial_data[ind][1], trial_data[ind][2])

    # Set x values for finalized model
    x_train_opt = x_train[:, [columns_list[ind][0], columns_list[ind][1], columns_list[ind][2]]]
    x_test_opt = x_test[:, [columns_list[ind][0], columns_list[ind][1], columns_list[ind][2]]]

    return x_train_opt, x_test_opt


def build_network(x_train, y_train, x_test, y_test, n_epoch=500):
    '''
    This function builds and trains the neural network.

    **Parameters**

        x_train: *numpy.ndarray*
            The domain of the training data. Should be a 2D array of 24 rows and three columns.
        y_train: *numpy.ndarray*
            The range of the training data. Should be a 1D array of length 24.
        x_test: *numpy.ndarray*
            The domain of the testing data. Should be a 2D array of 24 rows and three columns.
        y_test: *numpy.ndarray*
            The range of the testing data. Should be a 1D array of length 24.
        n_epoch: *int*
            Number of epochs for model. Default value of 100.

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
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # Add final layer, reducing # of nodes to 1 corresponding to estimated thermal decomposition temperature/100
    model.add(Dense(4))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=3, epochs=n_epoch, verbose=0, validation_data=(x_test, y_test))

    # Return the model
    return model, history


def evaluate_performance(model, history, x_test, y_test, x_vars, results):
    '''
    Retrieve MSE for a given set of parameters and append them to a list containing the MSE associated with the other
    other possible sets of parameters.

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

    # Retrieve loss and accuracy of the training data
    train_accuracy = history.history['accuracy'][len(history.epoch) - 1]
    train_loss = history.history['loss'][len(history.epoch) - 1]

    # Calculate loss and accuracy of the testing data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    results.append([x_vars[0], x_vars[1], x_vars[2], train_accuracy, test_accuracy])
    print(x_vars[0] + "+" + x_vars[1] + "+" + x_vars[2] + "\t \t \t" + str(round(train_loss, 3)) + "\t"
          + str(round(train_accuracy, 3)) + "\t" + str(round(test_accuracy, 3)))

    return results


def finalize_model(x_train, x_test, y_train, y_test, model_name='MOF_ThermalStability_NN.h5',
                   output_name='MOF_ChemicalStability_Plot.png', output_name2='MOF_ChemicalStability_Loss.png'):
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
            name of save file for plot of true and predicted chemical stability classification.
        output_name2: *str*
            name of save file for plots of loss and accuracy as a function of epoch.


    **Returns**

    '''

    # Generate model for chosen parameters
    model, history = build_network(x_train, y_train, x_test, y_test)

    # Generate predicted values with the improved model
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Combine and sort data sets
    train = []
    test = []
    for i, j in enumerate(y_train):
        train.append([mofs[i], j, float(train_pred[i])])

    for i, j in enumerate(y_test):
        test.append([mofs[i + len(y_train) - 1], j, float(test_pred[i])])

    train_sort = sorted(train, key=itemgetter(1))
    test_sort = sorted(test, key=itemgetter(1))

    # Plot predicted values as a function of actual values
    plt.figure(figsize=(5, 4))
    plt.subplot(2, 1, 1)
    plt.plot([z[1] for z in train_sort], 'r^')
    plt.plot([z[2] for z in train_sort], 'bo')
    x = list(range(len(y_train)))
    y = [0, 1, 2, 3]
    plt.yticks(y, ['Air Sensitive', 'Water Sensitive', 'Some Water Stability', 'Extensive Water Stability'])
    plt.xticks(x, [z[0] for z in train_sort], rotation=45)
    plt.title('Training Data')
    plt.ylabel('Chemical Stability Characterization')
    plt.legend(['True', 'Predicted'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot([z[1] for z in test_sort], 'r^')
    plt.plot([z[2] for z in test_sort], 'bo')
    y = list(range(len(y_test)))
    plt.ylim(0, 4)
    plt.xticks(y, [z[0] for z in test_sort], rotation=45)
    plt.title('Testing Data')
    plt.ylabel('Chemical Stability Characterization')
    plt.legend(['True', 'Predicted'], loc='lower right')
    plt.tight_layout()
    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(output_name)

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.tight_layout()
    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(output_name2)

    # Calculate loss and accuracy of the testing data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test Loss:", loss, "Test Accuracy:", accuracy)

    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load data into appropriate training and testing variables
    (x_train, y_train), (x_test, y_test), poss_params = load_data("WaterStability.xlsx")

    # Categorize y values
    n_classes = 4
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    # Determine which three parameters are best for predicting thermal decomposition temperature
    x_train_opt, x_test_opt = evaluate_xvars(x_train, y_train, x_test, y_test, poss_params)

    # Perform more extensive analysis of chosen parameters, save model, and produce plots of MSE and predicted values
    finalize_model(x_train_opt, x_test_opt, y_train, y_test)