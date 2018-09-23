from __future__ import print_function

import pandas as pd
import numpy as np
#create a series
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
#create a DataFrame
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(population)
print(cities['City name'])
# loa file in to dataframe 
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
# dataframe.describle() to show interesting statistics about a DataFrame
print(california_housing_dataframe.describe())
# useful function is DataFrame.head, which displays the first few records of a DataFrame
print(california_housing_dataframe.head())

# Another powerful feature of pandas is graphing. For example, DataFrame.hist lets you quickly study the distribution of values in a column
#print(california_housing_dataframe.hist('housing_median_age'))

# Accessing Data from dataFrame using python dict/list operation

#cities = pd.DataFrame({'City name' : city_names, 'Population' : population})
print(cities['City name'][1])
#cities[City name]

print(cities[0:2])
print("  ")
print(np.log(population))

# Exercise
#Modify the cities table by adding a new boolean column that is True if and only if both of the following are True:

#The city is named after a saint.
#The city has an area greater than 50 square miles.
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & (cities['City name'].apply(lambda name : name.startswith('San')))
print(cities)

# indexing
print(city_names.index)
print(cities.index)

# Call DataFrame.reindex to manually reorder the rows. For example, the following has the same effect as sorting by city name:
print(cities.reindex([2,0,1]))

# random.permutation function
print(cities.reindex(np.random.permutation(cities.index)))