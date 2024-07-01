"""
Let’s build a linear regression model for a real world example.
We are going to predict the house price based on its space features.
"""

# import all libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read file from file
file = "kc_house_data.csv"
df = pd.read_csv(file)
# print(df.head())

# THis prints large values with their exponents
# print(df.describe())

# See all columns names
# print(df.columns)

# Price us the target predicted - dependent
# Bedrooms,sqft living, yr_built are the predictors - independent
y = df["price"]
X = df[["bedrooms", "sqft_living", "yr_built"]]

# RAW DATA
sns.scatterplot(x=df["bedrooms"], y=y)
sns.scatterplot(x=df["sqft_living"], y=y)
sns.scatterplot(x=df["yr_built"], y=y)
plt.ticklabel_format(axis="y", style="plain")
plt.show()

"""
Note the mean house price is about $540,000, 
with a standard deviation of $367,000 so there is a broad range of house prices 
(68% are within 1 standard deviation,
 95% within 2 s.d. 
 and 99.7% within 3 s.d.):
"""
print("MEAN:", np.mean(y))
print("STD:", np.std(y))

"""
Building Model
Now, we can build the linear regression model. 
Formular: y=β0 +β1X1 +β2X2 +...+βpXp  == x values are input values whereas beta values are their coefficients.
We just want to know beta 1 to p coefficients and beta 0 intercept.
NOTE: This approach uses from sklearn.model_selection import train_test_split with
a corresponding train_size parameter (e.g. train_size=0.8 [same as 80% of population]) to denote the sample size
that should be randomly selected from the data to perform training of the model.
"""
from sklearn.model_selection import train_test_split

# Split the data from the train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Import Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression

# Fit the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the Salary for the test values
y_predictions = lr.predict(X_test)

# Intercept and coeff of the line
print("Intercept of the model:", lr.intercept_)
print("Coeff of the model:", lr.coef_)

"""
Our regression formula:
y=β0 +β1X1 +β2X2 +β3X3
Simplifying a bit (dropping everything to the right of the decimal):
y = 4601037 - 39547*X1 + 322*X2 - 2334*X3

Accuracy
The built lr regression model provides a predict function. 
We will call this function and pass features during training. 
We also have the actual values in the y dataframe. 
So we can calculate a Mean Absolute Error (mae) of our prediction.
sklearn comes with its own metrics as well and can calculate mae and other accuracy metrics.
"""
from sklearn import metrics
print("MEAN ABSOLUTE ERROR:", metrics.mean_absolute_error(y_test, y_predictions))
print("MEAN SQUARED ERROR:", metrics.mean_squared_error(y_test, y_predictions))
print("ROOT MEAN SQUARED ERROR:", np.sqrt(metrics.mean_squared_error(y_test, y_predictions)))

"""
Coefficient of determination
We can also calculate the R^2, otherwise known as the coefficient of determination.
 The coefficient of determination is an expression of the proportion of variation in the y
  variable that can be explained by variation in the features or X variable(s).
"""
R_squared = metrics.r2_score(y_test, y_predictions)
print("Coefficient of determination (R^2):", R_squared)  # output is ~ 0.529

"""
CONCLUSION
This means that 54% of the variation in house prices can be explained by the variation in our three features:
 bedrooms, square foot living space, and year built. 
 We can use this figure to compare with other combinations of features.
"""

#