import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.sav')
transformer = joblib.load('transformer.sav')

'''
THESE VALUES ARE CASE SENSITIVE

Gender: Male/Female
Partner: Yes/No
Dependents: Yes/No
PhoneService: Yes/No
MultipleLines: Yes/No/No phone service
InternetService: Fiber optic/DSL/No
OnlineSecurity: Yes/No/No internet service
OnlineBackup: Yes/No/No internet service
DeviceProtection: Yes/No/No internet service
TechSupport: Yes/No/No internet service
StreamingTV: Yes/No/No internet service
StreamingMovies: Yes/No/No internet service
Contract: Month-to-month/One year/Two year
SeniorCitizen: Yes/No -> 1/0
MonthlyCharges: (some float value)
TotalCharges: (some float value)

'''

gender = input('gender: ')
Partner = input('Partner: ')
Dependents = input('Dependents: ')
PhoneService = input('PhoneService: ')
MultipleLines = input('MultipleLines: ')
InternetService = input('InternetService: ')
OnlineSecurity = input('OnlineSecurity: ')
OnlineBackup = input('OnlineBackup: ')
DeviceProtection = input('DeviceProtection: ')
TechSupport = input('TechSupport: ')
StreamingTV = input('StreamingTV: ')
StreamingMovies = input('StreamingMovies: ')
Contract = input('Contract: ')
SeniorCitizen = int(bool(input('SeniorCitizen: '))) # 0 or 1
MonthlyCharges = float(input('MonthlyCharges: '))
TotalCharges = float(input('TotalCharges: '))



df = pd.DataFrame({'gender': [gender], 'Partner': [Partner], 'Dependents': [Dependents], 'PhoneService': [PhoneService], 'MultipleLines':[MultipleLines], 'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection], 'TechSupport':[TechSupport], 'StreamingTV':[StreamingTV], 'StreamingMovies':[StreamingMovies], 'Contract': [Contract], 'SeniorCitizen':[SeniorCitizen], 'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
df = df[['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']]
X = transformer.transform(df)
print(model.predict(X)[0])
