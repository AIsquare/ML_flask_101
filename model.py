# import pandas as pd # To manage data as data frames
# import numpy as np # To manipulate data as arrays
# from sklearn.linear_model import LogisticRegression

# def classify(a,b,c,d):
#     arr = np.array([a,b,c,d])
#     arr = arr.astype(np.float64)
#     query = arr.reshape(1,-1)
#     prediction = arr.reshape(1,-1)
#     #retrive form dictionary The probability score predicted using logistic regression can be used when finding the right mapping to the name of variety, using a dictionary called variety_mappings.
#     prediction = variety_mappings[logreg.predict(query)[0]]
#     return prediction

import joblib
#import matplotlib as plt
#import pandas as pd
#import numpy as np
import os
#from xgboost import XGBRegressor

#capture the of current folder
curr_path = os.path.dirname(os.path.realpath(__file__))

feat_cols = ['Distance','Haversine','Phour','Pmin','Dhour','Dmin','Temp',
             'Humid', 'Solar', 'Dust']
xgb_final = joblib.load(curr_path + "/final_model_best.joblib")

print(xgb_final)
def predict_duration(attributes: np.ndarray):
    '''Returns Biker Trip Duration value'''
    pred = xgb_final.predict(attributes)
    print("Duration predicted")
    return int(pred[0])
