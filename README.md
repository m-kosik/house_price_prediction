# house_price_prediction
a notebook for Kaggle competition about predicting house prices
  
This repository contains a notebook which allowed me to create a top-5% submission for [THIS](https://www.kaggle.com/c/home-data-for-ml-course) Kaggle competition.  
  
It contains a function for data preprocessing, which performs the following steps:  
- removing numerical columns which have more than `missing_threshold` missing entries,  
- imputing missing data in the rest of the numerical columns with missing data using the median value or 0,  
- imputing missing data in categorical columns with a new value (often the NaN in categorical data has the meaning of a new category, e.g. "the absence of a garage", therefore the imputation for categorical data is creating a new category where NaN is present),  
- removing features which have mutual information with the target smaller than `mutual_inf_threshold`,  
- creates new features by applying PCA to existing numerical features and then adding as new features the first `pca_components_to_include` components,  
- dropping categorical data that have cardinality larger than `low_cardinality_threshold`,  
- performing One Hot Encoding on the low-cardinality categorical data.  
  
Next, the preprocessed data is fit using a simple XGBRegressor model to find the optimal values of the parameters  `missing_threshold`, `mutual_inf_threshold`,  `pca_components_to_include`, `low_cardinality_threshold`. To save computation time, the XGBRegressor is run with early stopping rounds, when performed on the train/validations sets.  
  
Finally, a submission is created by running the XGBRegressor model on the data which was preprocessed with the optimal parameters.
