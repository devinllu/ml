# Processing Data

## Imputation

mean/median/mode imputation: 
- mode imputation is useful for categorical data
- median imputation is better for continuous data than mean (less sensitive to extremes where the mean is sensitive to)

KNN imputation - when the data exhibits a pattern/relationship, KNN can be useful for imputation but can be computationally costly for large datasets with tons of missing values. Best to scale/standardize the data beforehand to handle outliers beforehand

Regression imputation - developing a regression model and training it with data that does not have missing values as predictors for imputed values. Could consider multiple imputations where you repeat the process multiple times and average it

For MNAR data (missing not at random) - using a model based approach where you build a statistical model that accounts for the missing data or artificially creating data points based off the existing dataset
In this case, bayesian models seems to work well.

## Categorical Data Encoding

- one hot encoding (for nominal): creates a binary column for each categorical variable, removing ordinal relationships between variables but in doing so increases the feature space and could lead to overfitting / sparce matrices

- binary encoding (high cardinality): converts each categorical variable into a bit, which reduces dimensionality

- label encoding (for ordinal): converts each category into a unique integer, although might introduce an "ordering" that didn't exist prior (in the case of nominal data)
