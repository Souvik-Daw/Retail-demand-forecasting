import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from src.utils.exception import CustomException

def get_xgboost_preprocessor():
    """
    Pipeline for XGBoost (WITH LAGS).
    Features: [lags, rolling, day_of_week, month, year, onpromotion, store_nbr, family]
    """
    try:
        xgboost_cat_cols = ['family']
        # INCLUDE LAGS HERE
        xgboost_num_cols = [
            'lag_1', 'lag_7', 'lag_14', 
            'roll_7_mean', 'roll_14_mean', 
            'onpromotion', 'store_nbr', 'day_of_week', 'month', 'year'
        ]

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')) 
        ])

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('cat_trans', cat_pipeline, xgboost_cat_cols),
            ('num_trans', num_pipeline, xgboost_num_cols)
        ], remainder='drop') 

        return preprocessor
    except Exception as e:
        raise CustomException(e, sys)

def get_nn_preprocessor():
    """
    Pipeline for Neural Networks (NO LAGS).
    Features: [day_of_week, month, year, onpromotion, store_nbr, family]
    """
    try:
        # NO LAGS HERE
        nn_scale_cols = ['onpromotion']
        
        nn_cat_cols = ['family', 'store_nbr', 'month', 'day_of_week']
        nn_year_col = ['year']

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        year_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler())
        ])

        preprocessor = ColumnTransformer([
            ('num_trans', num_pipeline, nn_scale_cols),
            ('cat_trans', cat_pipeline, nn_cat_cols),
            ('year_trans', year_pipeline, nn_year_col)
        ], remainder='drop')

        return preprocessor
    except Exception as e:
        raise CustomException(e, sys)