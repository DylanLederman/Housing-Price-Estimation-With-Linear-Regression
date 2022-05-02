# Augmenta: House Prices Project

## Abstract

In this report, we created a linear regression model to predict the sale price of homes in the Ames Iowa Housing Dataset. First, we preprocessed the data by imputing missing values and encoding categorical values. Next, we applied feature transformation and hyperparameter optimization to improve the performance of our models. Then, we compared the performance of multiple models, including a Jax implementation of linear regression as well as XGBRegressor which uses gradient boosted random forests, and determined which models worked best under certain conditions. We concluded that XGBRegressor performed the best out of our models, while noting the effect of hyper-parameter tuning was less significant than that of feature engineering.

##Introduction 

The problem in focus is predicting the sale price of residential homes in Ames, Iowa based on their features including square footage, construction material, age, condition, and location among many others. This problem is of great interest since the complexity of the training data offers an array of preprocessing strategies, which significantly impact the performance of machine learning algorithms. In addition, although the data is complex, it can be learned by a variety of algorithms ranging over several levels of sophistication, which makes it a helpful tool for analyzing different implementations of linear regression.

## Related Work

1. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
2. https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
3. https://github.com/pantelis-classes/PRML/blob/master/prml/linear/_linear_regression.py
4. https://xgboost.readthedocs.io/en/stable/python/python_api.html
The paper Housing Price Prediction Based on Multiple Linear Regression by Qingqi Zhang attempts to predict house prices for a dataset of homes sold in Boston using multiple linear regression. Our approach will differ from this approach because we are using a single linear regression.
5. https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices
This introductory google collab was used as a reference in order to implement our data processing and test different encoding methods such as label, hot, and a combination of both
6. https://www.youtube.com/watch?v=aOsZdf9tiNQ&t=635s
This linear regression technique performed using jax was used as a reference in order to implement our linear regression using transformations of polynomial degree = 2 for testing

## Data

## Methods

## Experiments

## Conclusion 