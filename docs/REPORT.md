# Augmenta: House Prices Project

## Abstract

In this report, we created a linear regression model to predict the sale price of homes in the Ames Iowa Housing Dataset. First, we preprocessed the data by imputing missing values and encoding categorical values. Next, we applied feature transformation and hyperparameter optimization to improve the performance of our models. Then, we compared the performance of multiple models, including a Jax implementation of linear regression as well as XGBRegressor which uses gradient boosted random forests, and determined which models worked best under certain conditions. We concluded that XGBRegressor performed the best out of our models, while noting the effect of hyper-parameter tuning was less significant than that of feature engineering.

## Introduction 

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
7. In Incorporating Multiple Linear Regression in Predicting the House Prices Using a Big Real Estate Dataset with 80 Independent Variables, author Azad Adbulhafedh attempts to create a model with multiple regression that accurately predicts the prices of home in the Ames Housing Dataset. One of our approaches will be similar to this approach because we will utilize a random forest in XGBRegressor.

## Data

The data that we are working with is a collection of numeric values and categorical features describing an individual house in Ames, Iowa. Numeric features include Lot square footage, pool area, etc. Categorical features include house style, garage condition, etc. The dataset contains 79 features about a specific home, and 1460 rows of house data. The data was collected in 2011. 

The dataset needed preprocessing in order to be used for regression. Preprocessing of the dataset was broken up into three main parts: cleaning, categorizing, and handling missing values. 

During the cleaning phase of preprocessing, the description of the dataset was analyzed and compared to the actual data in order to check for any inconsistencies in our training data. Analysis revealed that the dataset contained some typos that differed from the specification of the data.

After fixing any typos or inconsistencies, the dataset was categorized. During this phase of the preprocessing, the categorical feature-set was split into nominative features and ordinal features. This process was done by-hand in order to increase the effectiveness of the encoding that will be done later. Examples of ordinal features 

In the final phase of preprocessing, the dataset was prepped to handle missing values. Developing a strategy for handling missing values requires understanding which features are missing values and how they should be treated. The following graph shows the top 6 features missing values.

###MissingValues
![missing values](https://github.com/hahdookin/cs301/blob/main/images/MissingValues.png)

After understanding why these features are missing values by reading their descriptions in the data description, appropriate action was taken to handle these missing values.

## Methods

To predict housing prices from the data set taken from Kaggle, we first process our data by imputing missing values and encoding categorical data types into numerical values. To achieve such, we take two separate approaches: label encoding and one-hot encoding. Label encoding involves assigning unique integers to a corresponding value of a categorical feature, whereas one-hot encoding splits each categorical feature’s possible values into separate columns and assigns them a binary value. Label encoding was applied to ordinal features and one-hot encoding was applied to nominative features.

After the data has been encoded we then transform it using the PolynomialFeature function from the pmrl library. A polynomial transformation was chosen to capture the nonlinear relationships between the model features. Once the data has been preprocessed, we tested three different linear regression models. The first model was a hand written implementation of jax following the tutorial from (https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html), the second was an implementation of the LinearRegression function from the pmrl library, and the third and best performing model was an implementation of XGBRegressor. 

## Experiments

Comparing rmse before and after hyperparameter tuning

Comparing rmse between different encoding methods such as one hot encoding, and label encoding, and a combination of the two

Because of the types of data present in our dataset, encoding methods need to be used in order for our regression to utilize the entire dataset. The following pie chart shows the distribution of data types present in our dataset. Where the columns correspond to the number of fields in the dataset.

###Data Distributions
![Data type distribution](https://github.com/hahdookin/cs301/blob/main/images/DataDistribution.png)

After viewing the pie chart, it is evident that over half of our dataset consists of non-numeric columns, which will not be used in a regression algorithm. It is clear that simply ignoring columns that contain categorical data will severely hinder the success of our model. Multiple 

encoding techniques were considered. After much consideration, a combination of label and one-hot encoding was used. Label encoding was applied to the ordinal features and one-hot encoding was applied to the nominative features. We then tested each encoding strategy with each model implementation, and recorded the performance of each model within a single encoding strategy (for figures below, “label” encoding was used). 

###Linear Regression Models
![Different Linear Regression Models](https://github.com/hahdookin/cs301/blob/main/images/DifferentMachineLearningResults.png)
####A visualization of predictions from three linear regression models plotted against the corresponding ground truth home sale prices. (Figures from left to right are from sources [1], [2], [3] respectively)

###Results of Encoding Techniques
![Encoding Techniques](https://github.com/hahdookin/cs301/blob/main/images/DifferentEncodingResults.png)

Comparing rmse between jax linear regression and xgboost

Comparing rmse between our implementation versus other kaggle


## Conclusion

Through the models created and different feature engineering methods, we concluded that XGBRegressor with feature-engineering performed the best, producing the lowest RMSE. The Jax implementation performed the worst of all of our models, but we would expect it to perform much better with more time to implement more sophisticated regression techniques. We learned that feature-engineering was far more important than hyper-parameter optimization, as it increased the effectiveness of XGBRegressor. We also learned about the importance of data encoding and feature-engineering. It was interesting to see how different encodings affected the results of our models.

Future improvements to solving this problem would include using a multiple linear regression model as well as more complex machine learning techniques such as ridge regression. Additionally, improving the capabilities of our Jax linear regression model would be done in the future. We would like to see the Jax model perform with a loss similar to that of the XGBRegressor.

Attempting this problem was enjoyable and served as a good introduction into applying a regression model to a real-world problem. 

