
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

import lightgbm as lgb


model_fname = "model.save"
MODEL_NAME = "text_class_base_lightgbm"


class Classifier(): 
    
    def __init__(self, **kwargs) -> None:
        self.model = self.build_model(**kwargs)     
        
        
    def build_model(self, **kwargs): 
        model = lgb.LGBMClassifier(
            **kwargs, 
            random_state=42
        )
        return model
    
    
    def fit(self, train_X, train_y, valid_X, valid_y):        
        self.model.fit(
            train_X, train_y,
            eval_set=[(valid_X, valid_y), (train_X, train_y)],
            verbose=20,
            )            
        
    
    def predict(self, X,): 
        preds = self.model.predict(X, num_iteration=self.model.best_iteration_)
        return preds 
    
    
    def predict_proba(self, X,): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)            

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):     
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model


def get_data_based_model_params(train_y, valid_y): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''      
    num_classes = len(set(train_y).union(set(valid_y)))
    if num_classes == 2:
        params = {
            "num_class": 1,  # weird thing about lightgbm that it requires num_class = 1 for binary classification
            "objective": "binary",
            "metric": "binary_logloss"
        }
    elif num_classes > 2:
        params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss"
        }
        
    return params