import csv
import pandas as pd
import random
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt


# https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
# https://stackoverflow.com/questions/43299500/pandas-how-to-know-if-its-day-or-night-using-timestamp/43299660
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.assign.html
# https://pandas.pydata.org/pandas-docs/stable/categorical.html


data = pd.read_csv("yellow_tripdata_2017-02.csv")
rowNum = data.shape[0]
dfSample = data.sample(10000)
print(dfSample.shape)
tip = dfSample['tip_amount']
fare = dfSample['fare_amount']
timestamp = dfSample['tpep_pickup_datetime']
paytype = dfSample['payment_type']
# print(type(paytype))
paytype_num = np.max(paytype)
# print(type(fare))

d = {}
for i in range(1, paytype_num + 1):
	d['payment_type_cat' + str(i)] = [0] * 10000



tip_rate=[]
dayornight = []
count = 0
# https://stackoverflow.com/questions/1060090/changing-variable-names-with-python-for-loops?lq=1
for idx, row in dfSample.iterrows():
	tip_rate.append(1 if tip[idx]/fare[idx] > 0.2 else 0)
	for i in range(1, paytype_num + 1):
		if paytype[idx] == i:
			d['payment_type_cat' + str(i)][count] = 1 
	count += 1
	# print(timestamp[idx])
	dayornight.append(1 if int(timestamp[idx][11:13]) < 19 and int(timestamp[idx][11:13]) >= 6 else 0)
	# print(1 if int(timestamp[idx][11:13]) < 19 and int(timestamp[idx][11:13]) >= 6 else 0)

	
dfSample=dfSample.assign(tip_rate_20 = tip_rate, DayorNight = dayornight)
for i in range(1, paytype_num + 1):
	# print(d['payment_type_cat' + str(i)])
	dfSample['payment_type_cat' + str(i)] = d['payment_type_cat' + str(i)]
	# print(dfSample['payment_type_cat' + str(i)])

# https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe
# https://stackoverflow.com/questions/30808430/how-to-select-columns-from-dataframe-by-regex
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.filter.html
# http://pandas.pydata.org/pandas-docs/stable/merging.html

dfP1 = dfSample.filter(items = ['VendorID', 'DayorNight', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID', 
	'tip_amount','fare_amount', 'tip_rate_20', 'payment_type'])
dfP2 = dfSample.filter(regex = '^payment_type_cat[0-9]$')
# print(dfP2)
frames = [dfP1, dfP2]

result = pd.concat(frames, axis=1)
result.to_csv("processed_trip_data.csv")



x = tip.values
y = fare.values
plt.scatter(x.tolist(), y.tolist(), s = 5)
plt.xlabel("tip amount")
plt.ylabel("fare amount")
plt.title("tip & fare Distribution")
plt.show()






