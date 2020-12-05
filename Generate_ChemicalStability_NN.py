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
from operator import itemgetter


def load_data(fptr, columns=[], n_training=32):
    '''
    Loads data from an Excel file containing possible parameters (Linker functional group classification, linker pKa,
    Metal d electron count, metal electronegativity, metal charge in MOF complex, linker molecular weight, number of
    metal atoms bound to a single linker molecule, and Langmuir surface area) to use as inputs to NN, MOF names to use
    as labels, and chemical stability classification for MOFs used as outputs.

     **Parameters**

        fptr: *str*
            name of file containing data.
        columns: *list*
            allows a user to specify three possible parameters from those listed above to be extracted from the Excel
            file. This allows another python script to access this function to use with a pre-existing model.
        n_training: *int*
            number of training data points to be used from dataset.

    **Returns**
        x_train_tot: *numpy.ndarray*
            array containing domain of training data set. Should be a 2D array of 24 rows and three columns.
        y_train: *numpy.ndarray*
            array containing range of training data set. Should be a 1D array of length 24.
        x_test_tot: *numpy.ndarray*
            array containing domain of testing data set. Should be a 2D array of three columns.
        y_test: *numpy.ndarray*
            array containing range of testing data set. Should be a 1D array.

    '''

    assert os.path.exists(fptr), "File should be in linked directory"

    book = xlrd.open_workbook(fptr)

    dataset = book.sheet_by_index(0)
    book.release_resources()
    num_MOFs = dataset.nrows - 1

    assert num_MOFs >= 32, "Number of MOFs should be at least 32"
    assert dataset.ncols - 5 == 8, "Number of Possible Inputs Should be 8"

    # Initialize Matrices
    data = np.empty((num_MOFs, 9))
    params = []
    mofs = []

    # If statement allowing specified columns to be chosen
    if columns:
        for j in range(1, num_MOFs + 1):
            mofs.append(dataset.cell_value(j, 0))
            for i in columns:
                data[j - 1, i - 4] = dataset.cell_value(j, i)
                params.append(dataset.cell_value(0, i))
                data[j - 1, 3] = dataset.cell_value(j, 12)
        print("Selected Parameters:", params[0], params[1], params[2])
        x_train = data[0:n_training, 0:3]
        y_train = data[0:n_training, 3]
        x_test = data[n_training:num_MOFs, 0:3]
        y_test = data[n_training:num_MOFs, 3]

        return x_train, y_train, x_test, y_test, mofs

    else:
        # Move Data from Excel File (x already normalized and MOFs randomized in Excel) into Array
        for j in range(1, num_MOFs + 1):
            mofs.append(dataset.cell_value(j, 0))
            for i in range(4, dataset.ncols):
                data[j - 1, i - 4] = dataset.cell_value(j, i)

        for i in range(4, dataset.ncols - 1):
            params.append(dataset.cell_value(0, i))

        # Separate Training and Testing Data Sets
        x_train_tot = data[0:n_training, 0:8]
        y_train = data[0:n_training, 8]
        x_test_tot = data[n_training:num_MOFs, 0:8]
        y_test = data[n_training:num_MOFs, 8]

        return (x_train_tot, y_train), (x_test_tot, y_test), params, mofs


def evaluate_inputs(x_train_tot, y_train, x_test_tot, y_test, params, filename='MOF_ChemicalStability_NN.txt'):
    '''

     **Parameters**
        x_train_tot: *numpy.ndarray*
            array containing domain of training data set
        y_train: *numpy.ndarray*
            array containing range of training data set
        x_test_tot: *numpy.ndarray*
            array containing domain of testing data set
        y_test: *numpy.ndarray*
            array containing range of testing data set
        params: *numpy.ndarray*
            1D array of possible linker, metal, and MOF characteristics from dataset

    **Returns**
        x_train_opt: *numpy.ndarray*
            array containing training x data of the three parameters that contributed to the smallest loss when used to
            generate the model. Should be a 2D array of 24 rows and three columns.
        x_test_opt
            array containing testing x data of the three parameters that contributed to the smallest loss when used to
            generate the model. Should be a 2D array of 14 rows and three columns.

    '''

    assert filename[-4:] == '.txt', "Save file should be a text file"

    num_params = list(range(0, len(params)))
    poss_combos = combinations(num_params, 3)
    results = []
    columns_list = []
    print("X Variables \t \t Training Loss \t Training Accuracy \t Testing Loss \t Testing Accuracy")

    for i in poss_combos:
        x_train = x_train_tot[:, [i[0], i[1], i[2]]]
        x_test = x_test_tot[:, [i[0], i[1], i[2]]]
        # List of Linker, Metal, or MOF properties evaluated in current iteration
        inputs = [params[i[0]], params[i[1]], params[i[2]]]
        columns_list.append([i[0], i[1], i[2]])
        # Build network
        model, history = build_network(x_train, y_train, x_test, y_test, 1)
        # Observe the performance of the network
        trial_data = evaluate_performance(model, history, x_test, y_test, inputs, results)

    # Write .txt file containing columns for each of the three inputs for a given iteration, and the loss for the
    # training and testing data sets.
    data_file = open(filename, 'w')
    for line in trial_data:
        data_file.write(str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\t" + str(line[3]) + "\t" + str(line[4]) + "\n")
    data_file.close()

    # Return index of best parameters to approximate the system. Best parameters judged by smallest mean squared error
    # of the model on the testing data.
    combined_loss = [float(x[3]) + float(x[4]) for x in trial_data]
    ind = combined_loss.index(min(combined_loss))
    print('Chosen parameters for optimized model :', trial_data[ind][0], trial_data[ind][1], trial_data[ind][2])

    # Set x values for finalized model
    x_train_opt = x_train_tot[:, [columns_list[ind][0], columns_list[ind][1], columns_list[ind][2]]]
    x_test_opt = x_test_tot[:, [columns_list[ind][0], columns_list[ind][1], columns_list[ind][2]]]

    return x_train_opt, x_test_opt


def build_network(x_train, y_train, x_test, y_test, n_epoch=300):
    '''
    This function builds and trains the neural network.

    **Parameters**

        x_train: *numpy.ndarray*
            The domain of the training data. Should be a 2D array of 24 rows and three columns.
        y_train: *numpy.ndarray*
            The range of the training data. Should be a 1D array of length 24.
        x_test: *numpy.ndarray*
            The domain of the testing data. Should be a 2D array with three columns.
        y_test: *numpy.ndarray*
            The range of the testing data. Should be a 1D array.
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
    model.add(Dense(7, input_shape=(3,), activation='relu'))
    model.add(Dropout(0.5))
    # Add second layer. Don't have to specify shape any more.
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.5))
    # Add final layer, reducing # of nodes to 1 corresponding to predicted chemical stability classification
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=1, epochs=n_epoch, verbose=0, validation_data=(x_test, y_test))

    # Return the model
    return model, history


def evaluate_performance(model, history, x_test, y_test, inputs, results):
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
        inputs: *list*
            List containing set of parameters used as inputs for current iteration of the model.
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
    results.append([inputs[0], inputs[1], inputs[2], train_loss, test_loss])
    print(inputs[0] + "+" + inputs[1] + "+" + inputs[2] + "\t \t \t" + str(round(train_loss, 3)) + "\t \t \t \t"
          + str(round(train_accuracy, 3)) + "\t \t \t \t" + str(round(test_loss, 3)) + "\t \t \t \t" +
          str(round(test_accuracy, 3)))

    return results


def finalize_model(x_train, x_test, y_train, y_test, model_name='MOF_ChemicalStability_NN.h5',
                   loss_plot='MOF_ChemicalStability_Loss.png', accuracy_plot='MOF_ChemicalStability_Accuracy.png'):
    '''
    This function builds and trains a neural network using the optimal combination of parameters. It then plots loss of
    the model for the training and testing data sets as a function of the epoch. The function also saves the plot and
    the model.

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
        loss_plot: *str*
            name of save file for plot of loss as a function of epoch for training and testing data.
        accuracy_plot: *str*
            name of save file for plot of accuracy as a function of epoch for training and testing data.

    **Returns**

        model:
            A class object that holds the trained model.


    '''

    assert model_name[-3:] == '.h5', "Model should be saved as a .h5 file"
    assert loss_plot[-4:] == accuracy_plot[-4:] == '.png', "Figures should be saved as a .png file"

    # Generate model for chosen parameters
    model, history = build_network(x_train, y_train, x_test, y_test, 2000)
    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    # Calculate loss and accuracy of the testing data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test Loss:", loss, "Test Accuracy:", accuracy)

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.tight_layout()
    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(loss_plot)

    # Plot Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.tight_layout()
    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(accuracy_plot)

    return model


def plot_model(x_train, y_train, x_test, y_test, mofs, model, train_plot='MOF_ChemicalStability_Train.png',
               test_plot='MOF_ChemicalStability_Test.png'):
    '''
    This function plots the true and predicted chemical stability classification values for the training and testing
    data sets. The function then saves both plots.

    **Parameters**
        x_train: *numpy.ndarray*
            The domain of the training data. Should be a 2D array of 24 rows and three columns.
        y_train: *numpy.ndarray*
            The range of the training data. Should be a 1D array of length 24.
        x_test: *numpy.ndarray*
            The domain of the testing data. Should be a 2D array of 24 rows and three columns.
        y_test: *numpy.ndarray*
            The range of the testing data. Should be a 1D array of length 24.
        model:
            A class object that holds the trained model.
        train_plot: *str*
            name of save file for plot of true and predicted chemical stability classification.
        test_plot: *str*
            name of save file for plot of true and predicted chemical stability classification.

    **Returns**

    '''

    assert train_plot[-4:] == test_plot[-4:] == '.png', "Figures should be saved as a .png file"

    assert len(x_train[0]) == len(x_test[0]) == 3, "Should only have three inputs for plot_model"
    assert len(x_train) >= 24, "Should have data for at least 24 MOFs in x_train"
    assert len(x_test) >= 8, "Should have data for at least 8 MOFs in x_test"

    # Generate predicted values with the improved model
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_pred = np.argmax(train_pred, 1)
    test_pred = np.argmax(test_pred, 1)

    # Combine and sort data sets
    train = []
    test = []
    for i, j in enumerate(y_train):
        train.append([mofs[i], np.argmax(j), float(train_pred[i])])

    for i, j in enumerate(y_test):
        test.append([mofs[i + len(y_train) - 1], np.argmax(j), float(test_pred[i])])

    train_sort = sorted(train, key=itemgetter(1))
    test_sort = sorted(test, key=itemgetter(1))

    # Plot predicted values as a function of actual values
    plt.figure()
    plt.plot([z[1] for z in train_sort], 'r^')
    plt.plot([z[2] for z in train_sort], 'bo')
    x = list(range(len(y_train)))
    y = [0, 1, 2, 3]
    plt.yticks(y, ['Air Sensitive', 'Water Sensitive', 'Some Water Stability', 'Extensive Water Stability'])
    plt.xticks(x, [z[0] for z in train_sort], rotation=45, fontsize=6)
    plt.title('Training Data')
    plt.ylabel('Chemical Stability Characterization')
    plt.legend(['True', 'Predicted'], loc='lower right')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(train_plot)

    plt.figure()
    plt.plot([z[1] for z in test_sort], 'r^')
    plt.plot([z[2] for z in test_sort], 'bo')
    y = [0, 1, 2, 3]
    plt.yticks(y, ['Air Sensitive', 'Water Sensitive', 'Some Water Stability', 'Extensive Water Stability'])
    y = list(range(len(y_test)))
    plt.xticks(y, [z[0] for z in test_sort], rotation=45, fontsize=7)
    plt.title('Testing Data')
    plt.ylabel('Chemical Stability Characterization')
    plt.legend(['True', 'Predicted'], loc='lower right')
    plt.tight_layout()
    fig = plt.gcf()
    # Save figure as .png file
    fig.savefig(test_plot)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load data into appropriate training and testing variables
    (x_train, y_train), (x_test, y_test), params, mofs = load_data("WaterStability.xlsx")

    # Categorize y values
    n_classes = 4
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    # Determine which three parameters are best for predicting chemical stability classification
    x_train_opt, x_test_opt = evaluate_inputs(x_train, y_train, x_test, y_test, params)

    # Perform more extensive analysis of chosen parameters and save model
    model = finalize_model(x_train_opt, x_test_opt, y_train, y_test)

    # Prepare plots of finalized model
    plot_model(x_train_opt, y_train, x_test_opt, y_test, mofs, model)
