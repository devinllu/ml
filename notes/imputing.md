Handling Missing Data

mean/median/mode imputation: 
- mode imputation is useful for categorical data
- median imputation is better for continuous data than mean (less sensitive to extremes where the mean is sensitive to)

KNN imputation - when the data exhibits a pattern/relationship, KNN can be useful for imputation but can be computationally costly for large datasets with tons of missing values. Best to scale/standardize the data beforehand to handle outliers beforehand

Regression imputation - developing a regression model and training it with data that does not have missing values as predictors for imputed values. Could consider multiple imputations where you repeat the process multiple times and average it

For MNAR data (missing not at random) - using a model based approach where you build a statistical model that accounts for the missing data or artificially creating data points based off the existing dataset
In this case, bayesian models seems to work well.

