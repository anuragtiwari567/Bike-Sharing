# Bike sharing demand prediction using hourly dataset

# Importing necessary data processing libraries
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# Loading Data
data = pd.read_csv("hour.csv")
bikes = data.copy()

# Dropping the non essential features
bikes.drop(['index','date','casual','registered'],axis = 1,inplace=True)

# Basic Data Checks

## Null values ?
bikes.isnull().sum()

# Visualize the data using pandas histogram...
bikes.hist(rwidth=0.6)
plt.tight_layout()

# Demand is not normally distributed - predicted variable

## Visualize the data for continuos features
plt.subplot(2,2,1)
plt.title("Temp vs Demand")
plt.scatter(bikes.temp,bikes.demand,s=2,c='g')

plt.subplot(2,2,2)
plt.title("atemp vs Demand")
plt.scatter(bikes.atemp,bikes.demand,s=2,c='c')

plt.subplot(2,2,3)
plt.title("humidity vs Demand")
plt.scatter(bikes.humidity,bikes.demand,s=2,c='r')

plt.subplot(2,2,4)
plt.title("windspeed vs Demand")
plt.scatter(bikes.windspeed,bikes.demand,s=2,c='b')

plt.tight_layout()

# Higher the windspeed, lower the demand

# Visualizing the categorical data
# Finding average demand for each category and plotting
cat_variables = ['season','year','month','hour','holiday','weekday','workingday','weather']

for cat in cat_variables:
    cat1_x = bikes[cat].unique()
    cat1_y = bikes.groupby(cat).mean()['demand']
    plt.subplot(3,3,cat_variables.index(cat)+1)
    plt.title("Avg demand/{}".format(cat))
    plt.xlabel("{}".format(cat))
    plt.ylabel("Avg demand")
    colors = ['b','r','m','g']
    plt.bar(cat1_x,cat1_y,color=colors)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=1,wspace=2)

# WorkingDay doesnt affect the demand as such

# Separate plot for visualizing the hourly trend
cat1_x = bikes['hour'].unique()
cat1_y = bikes.groupby('hour').mean()['demand']
plt.title("Avg demand/hour")
plt.xlabel("hour")
plt.ylabel("Avg demand")
colors = ['b','r','m','g']
plt.bar(cat1_x,cat1_y,color=colors)

# Findings-
# The demand is not normally distibuted therefore tranformation reqd.
# Temperature and Demand seems to be directly correlated
# Plots of temp and atemp are almost identical pointing out to some correlation, therefore a multicollinearity check is reqd.
# Humidity and Windspeed affects demand but need more statistical analysis like correlation coefficient check
# Park bikes near public transport in morning and office premises in the afternoon
# Weekday doesn't affect the demand therefore can be dropped
# Year doesnt affect since only 2 years given

# # Checking Outliers

bikes.demand.describe()
# 50% of the data is between 40 and 281 and fairly away from max and min
# Use Pandas quantile function to define the threshold for outliers
bikes.demand.quantile([0.05,0.1,0.15,0.90,0.95,0.99])
# 5% of the time the demand is equal to or less than 5 bikes
# 1% of the time it is equal to or above 782 bikes

# # Checking Multiple Linear Regression Assumptions
# ------Check Linearity using correlation coefficient matrix using corr -----

# Continuous Variables :
correlation = bikes[['temp','atemp','humidity','windspeed','demand']].corr()
# Drop a temp since showing high multicollinearity with temp
# Also humidity has a high correlation with wind speed. And windspeed has low Correlation with demand. 
# Therefore windspeed could be dropped

bikes.drop(['weekday','year','workingday','atemp','windspeed'],axis = 1,inplace=True)

## Checking auto-correlation with the demand :
plt.acorr(bikes.demand.astype('float'),maxlags=12)
# High auto-correlation upto 5 previous values(Top 3 > 0.8)
# Since autocorrelation is in the dependent variable, we cant get rid of it

# # # Solving the problem of Normality

#Since demand is a log normal distribution we will use the log transformation to get Normal distribution

df = bikes.demand
df2 = np.log(df)

plt.figure()
df.hist(rwidth=0.9,bins=20)

plt.figure()
df2.hist(rwidth=0.9,bins=20)

bikes['demand'] = df2 

# # #  Taking Care of AutoCorrelation :
    
t_1 = bikes.demand.shift(+1).to_frame()
t_1.columns = ['t-1']
t_2 = bikes.demand.shift(+2).to_frame()
t_2.columns = ['t-2']
t_3 = bikes.demand.shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes,t_1,t_2,t_3],axis =1)
bikes_prep_lag = bikes_prep_lag.dropna()
# Therefore three additional columns are also used to predict the demand--> Time lag cols will decrease the autocorrelation

# # Changing the categorical variables to dummy variables(make type category):

cat_variables_final = ['season','month','hour','holiday','weather']
bikes_prep_lag[cat_variables_final] = bikes_prep_lag[cat_variables_final].astype('category')
dummy_df = pd.get_dummies(bikes_prep_lag,drop_first=True)

# # Splitting the data

# Since its a Time dependent data we cant choose random elements as train and test since the data elements are correlated

Y = dummy_df['demand']
X = dummy_df.drop(['demand'],axis=1)

train_size = int(0.7 * len(dummy_df))

X_train = X.values[0:train_size]
X_test = X.values[train_size:len(X)]

Y_train = Y.values[0:train_size]
Y_test = Y.values[train_size:len(X)]

# # Prediction using multiple linear regression

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,Y_train)

# R squared values

r2_train = linreg.score(X_train,Y_train)
r2_test = linreg.score(X_test,Y_test)

# Making Predictions

Y_predict = linreg.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test,Y_predict))

# # # Calculate the RMSLE for Kaggle

# Since RMSLE utilizes the log values and we already have log values for Y, first convert to exponential values

Y_test_e = []
Y_predict_e = []
log_squared_sum = 0.0

for i in range(0,len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
# Do the sum of logs and squares :
    
for i in range(0,len(Y_test_e)):
    log_a = math.log(Y_test_e[i]+1)
    log_p = math.log(Y_predict_e[i]+1)
    log_diff = (log_p - log_a)**2
    log_squared_sum = log_squared_sum + log_diff
    
RMSLE = math.sqrt(log_squared_sum/len(Y_test))

print("RMSLE Score for the current model is -> {}".format(RMSLE))
# # # RMSLE = 0.3560. We fall under the top 1 percentile(<0.367) of the Kaggle Bike Prediction Comp.
