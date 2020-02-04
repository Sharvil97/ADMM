# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 02:44:12 2019

@author: Chanakya-vc
"""

from Admm_Lasso_solver import ADMM
from sklearn import datasets, linear_model
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# =============================================================================
# Choose only one feature from the dataset
# =============================================================================
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]

# =============================================================================
 # Split the data into training/testing sets
# =============================================================================
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# =============================================================================
#  Split the targets into training/testing sets
# =============================================================================
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]



# Create linear regression object
# regr = linear_model.Lasso()
regr = linear_model.Ridge()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# =============================================================================
# Call the ADMM solver for Lasso problem
# =============================================================================
A = diabetes_X_train
Y = diabetes_y_train
b = np.ones((A.shape[0], 1))
A = np.concatenate([A , b], axis = 1)
Y = Y.reshape(-1, 1)
tau = np.linspace(0.001, 0.1, num=10)
for tau in tau:
	model=ADMM(A,Y,tau)
	model.get_X(100,"lasso")
	Y_pred=model.predict(diabetes_X_test)

	# =============================================================================
	#  Plot outputs
	# =============================================================================
	# plt.figure(figsize=(10,6))
	# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black',label='Actual')
	# plt.plot(diabetes_X_test, Y_pred, color='blue', linewidth=1,label='Lasso Prediction by ADMM')
	# plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=1, label = 'Prediction by SK-LEARN') #sklearn comparison
	# plt.title('ADMM LASSO vs SK-LEARN LASSO')
	# plt.xlabel("Features")
	# plt.ylabel("Y_labels")
	# plt.xticks(())
	# plt.yticks(())
	# plt.legend()
	# plt.show()


	# =============================================================================
	# Call the ADMM solver for ridge problem
	# =============================================================================
	A = diabetes_X_train
	Y = diabetes_y_train
	b = np.ones((A.shape[0], 1))
	A = np.concatenate([A , b], axis = 1)
	Y = Y.reshape(-1, 1)
	model=ADMM(A,Y,0.1)
	model.get_X(100,"ridge")
	Y_pred=model.predict(diabetes_X_test)

	# =============================================================================
	#  Plot outputs
	# =============================================================================
	plt.figure(figsize=(10,6))
	plt.scatter(diabetes_X_test, diabetes_y_test,  color='black',label='Actual')
	plt.plot(diabetes_X_test, Y_pred, color='blue', linewidth=1,label='Ridge Prediction by ADMM')
	plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=1, label = 'Prediction by SK-LEARN') #sklearn comparison
	plt.title('ADMM Ridge vs SK-LEARN Ridge')
	plt.xlabel("Features")
	plt.ylabel("Y_labels")
	plt.xticks(())
	plt.yticks(())
	plt.legend()
	plt.show()