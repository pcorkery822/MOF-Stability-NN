'''
Use Generate_ChemicalStability_NN.py file to load data from Excel file and load model MOF_ChemicalStability_NN.h5 to
make plots of predicted and true values of chemical stability classification of MOFs in loaded data.
 '''

import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from keras.models import load_model
from Generate_ChemicalStability_NN import load_data, plot_model

if __name__ == '__main__':
	# Retrieve Data for Selected Columns from Excel File
	x_train, y_train, x_test, y_test, mofs = load_data('WaterStability.xlsx', columns=[4, 5, 6])

	# Categorize y values
	n_classes = 4
	y_train = np_utils.to_categorical(y_train, n_classes)
	y_test = np_utils.to_categorical(y_test, n_classes)

	# Load Pre-Existing Model
	model = load_model('MOF_ChemicalStability_NN.h5')

	# Prepare Plots of Finalized Model
	plot_model(x_train, y_train, x_test, y_test, mofs, model, "Chemical Stability Training.png",
			   "Chemical Stability Testing.png")

