import pickle
import sys
import os
import pandas as pd
# from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_recall_curve

inpu=pd.read_csv("now.csv")
df=pd.DataFrame(inpu)


loaded_model = pickle.load(open('./outputs/model.pkl','rb'))
result = loaded_model.predict(df)
print(pd.DataFrame(result))