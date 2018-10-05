import csv
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# Sample 1000 data
data = pd.read_csv("../data/processed_trip_data.csv")
data = data.sample(1000)

# Use attribute b, c, d, h as input features for KNN model
def get_input_matrix_knn(df):
	df1 = df.filter(items = ['DayorNight', 'passenger_count', 'trip_distance'])
	df2 = df.filter(regex = '^payment_type_cat[0-9]$')  
	print(df1.shape)
	print(df2.shape)
	frames = [df1, df2]
	return pd.concat(frames, axis=1)

# Use attribute k as class labels.
def get_output_matrix(df):
	return df['tip_rate_20']

# Build decision tree with attribute b, c, d, g.
def get_input_matrix_dt(df):
	return df.filter(items = ['DayorNight', 'passenger_count', 'payment_type', 'trip_distance'])


# KNN
taxi_X_train_knn = get_input_matrix_knn(data)
taxi_Y_train = get_output_matrix(data)


Kneigh = KNeighborsClassifier(n_neighbors=5, p=2)
predicted = cross_val_predict(Kneigh, taxi_X_train_knn, taxi_Y_train, cv=5)
print("========KNeighborsClassifier==========")
print(metrics.classification_report(predicted, taxi_Y_train))


# Decision Tree
print("========DecisionTreeClassifier==========")
taxi_X_train_dt = get_input_matrix_dt(data)
# For trip distance, you can calculate the average and use it as the threshold to create conditions.
distMean = taxi_X_train_dt['trip_distance'].mean()
for idx, row in taxi_X_train_dt.iterrows():
	if taxi_X_train_dt['trip_distance'][idx] < distMean:
		taxi_X_train_dt['trip_distance'][idx] = 0
	else:

		taxi_X_train_dt['trip_distance'][idx] = 1

decisionTree = DecisionTreeClassifier(random_state=0)

predicted = cross_val_predict(decisionTree, taxi_X_train_dt, taxi_Y_train, cv=5)
print(metrics.classification_report(predicted, taxi_Y_train))