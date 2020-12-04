'''
Use Generate_ThermalStability_NN.py file to load data from Excel file and load model MOF_ThermalStability_NN.h5 to
make plots of predicted and true values of thermal decomposition temperature of MOFs in loaded data.
 '''

import numpy as np
from tensorflow import keras
from keras.models import load_model
from Generate_ThermalStability_NN import load_data, plot_model

if __name__ == '__main__':
	# Retrieve Data for Selected Columns from Excel File
	x_train, y_train, x_test, y_test, mofs = load_data('ThermalStability.xlsx', columns=[4, 5, 6])

	# Load Pre-Existing Model
	model = load_model('MOF_ThermalStability_NN.h5')

	# Prepare Plots of Finalized Model
	plot_model(x_train, y_train, x_test, y_test, mofs, model, "Thermal Stability Training.png",
			   "Thermal Stability Testing.png")
