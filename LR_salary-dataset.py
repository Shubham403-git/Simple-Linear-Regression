import pandas
dataset = pandas.read_csv('SalaryData.csv')
y = dataset['Salary']
x = dataset['YearsExperience']
x = x.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
# linear function : y = b + cx : model
model = LinearRegression()
# model training
model.fit(x,y)