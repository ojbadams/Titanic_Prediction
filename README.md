# Titanic_Prediction
Entry for https://www.kaggle.com/c/titanic


## Methodology
- We're missing data for several columns, we want to use as much data as possible so we'll combine both train and test to build models to impute missing data. 
- Currently, there are only very basic models to impute data. 
- Once the missing data is imputed we then run the model through a predictor.

### Layout
- *titanic.ipynb* analysis of data.
- *data_processing.py* classes for imputation and sklearn pipeline.

## Results
For the test set Kaggle returned a result of 0.76076.

The principle here is provide a baseline by using simplistic ML models. Before advancing to more coomplex ML models we'll need to change some of the feature engineering.

## Next Steps
- Make better use of features (titles etc)
- Experiment with more advanced models (NN, XGBoost, etc)