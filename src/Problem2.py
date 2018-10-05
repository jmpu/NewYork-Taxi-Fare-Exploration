import csv
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import statistics
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
data = pd.read_csv("../data/processed_trip_data.csv")

def get_input_matrix(df):
	df1 = df.filter(items = ['DayorNight', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID'])
	df2 = df.filter(regex = '^payment_type_cat[0-9]$')
	print(df1.shape)
	print(df2.shape)
	frames = [df1, df2]
	return pd.concat(frames, axis=1)

def get_output_matrix(df):
	return df['fare_amount']

taxi_X_train = get_input_matrix(data)
taxi_Y_train = get_output_matrix(data)
print(taxi_X_train.shape)
print(taxi_Y_train.shape)

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
lr = linear_model.LinearRegression()

# Here we use cross_val_score for cross validation, set scoring as 'neg_mean_squared_error'
# So that we can obtain scores which are actually negative MSE for each fold.
# By -np.mean(scores) and np.std(scores), we obtain the averaged MSE and its std for all 5 folds.
scores = cross_val_score(lr, taxi_X_train, taxi_Y_train, scoring = 'neg_mean_squared_error',cv=5)
print("========LinearRegression=========")
print("MSE scores for 5 folds: ", [-score for score in scores])
print("Averaged MSE score - [LinearRegression]: ", - np.mean(scores))
std_score = np.std(scores)
print("Standard Deviation of MSE scores for 5 folds - [LinearRegression]:", std_score)
# predicted = cross_val_predict(lr, taxi_X_train, taxi_Y_train, cv=5)
# print(mean_squared_error(predicted, taxi_Y_train))


# Use a for loop to iterate through k from 1 to 10, 
# choose the k that obtains the minimum MEAN SQUARE ERROR, which indicates the best model.
print("=========KNNRegression=========")
mse_min = 0; k = 0
for i in range(1, 11):
	neigh = KNeighborsRegressor(n_neighbors=i)
	scores = cross_val_score(neigh, taxi_X_train, taxi_Y_train, scoring = 'neg_mean_squared_error', cv=5)
	mean_score = - np.mean(scores)
	if mse_min == 0 or mse_min > mean_score:
		mse_min = mean_score
		k = i
print("The opitimal K value for the KNN regression model: ", k, ", which obtains MINIMUM mse score: ", mse_min)


neigh = KNeighborsRegressor(n_neighbors=2)
scores = cross_val_score(neigh, taxi_X_train, taxi_Y_train, scoring = 'neg_mean_squared_error', cv=5)
mean_score = - np.mean(scores)
print("Averaged MSE score - [KNN regression]: ", mean_score)
std_score = np.std(scores)
print("Standard Deviation of MSE scores for 5 folds - [KNN regression]:", std_score)






