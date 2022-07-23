import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_player import st_player

import streamlit.components.v1 as components
model = joblib.load('model.sav')
transformer = joblib.load('transformer.sav')
st.set_page_config(layout="wide")
with st.sidebar:
    
    choose = option_menu("Welcome", ["Home", "Tech Stack","Predictor","ML Code", "Contributors"],
                         icons=['house', 'stack', 'robot','terminal', 'people-fill'],
                         menu_icon="clipboard-data", default_index=0, 
                         styles={
                            "container": {"padding": "5!important", "background-color": "#1a1a1a"},
                            "icon": {"color": "White", "font-size": "25px"}, 
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4d4d4d"},
                            "nav-link-selected": {"background-color": "#4d4d4d"},
                        }
    ) 


with open("contributors.html",'r') as f:
   contributors=f.read();
def html():
    components.html(
      contributors
     ,
    height=1400,
    
    scrolling=True,
)
def pred():
    st.title("TELECOM CUSTOMER CHURN MODELLING")

    Gender = st.selectbox(" ", ('Male', 'Female'))
    st.write('Gender')
    Partner = st.selectbox("  ", ('Yes', 'No'))
    st.write('Partner')
    Dependents = st.selectbox("   ", ('Yes', 'No'))
    st.write('Dependents')
    PhoneService = st.selectbox("    ", ('Yes', 'No'))
    st.write('PhoneService')
    MultipleLines = st.selectbox("     ", ('Yes', 'No','No phone service'))
    st.write('MultipleLines')
    InternetService = st.selectbox("      ", ('Fiber optic','DSL', 'No'))
    st.write('InternetService')
    OnlineSecurity = st.selectbox("        ", ('Yes', 'No','No internet service'))
    st.write('OnlineSecurity')
    OnlineBackup = st.selectbox("                  ", ('Yes', 'No','No internet service'))
    st.write('OnlineBackup')
    DeviceProtection = st.selectbox("                   ", ('Yes', 'No','No internet service'))
    st.write('DeviceProtection')
    TechSupport = st.selectbox("                              ", ('Yes', 'No','No internet service'))
    st.write('TechSupport')
    StreamingTV = st.selectbox("                                    ", ('Yes', 'No','No internet service'))
    st.write('StreamingTV')
    StreamingMovies = st.selectbox("                                   ", ('Yes', 'No','No internet service'))
    st.write('StreamingMovies')
    Contract = st.selectbox("                                                 ", ('Month-to-month','One year','Two year'))
    st.write('')
    SeniorCitizen = st.selectbox("                 ", ('Yes', 'No'))
    st.write('SeniorCitizen')
    if SeniorCitizen=='Yes':
        SeniorCitizen=1
    else:
        SeniorCitizen=0
    MonthlyCharges = st.number_input(' ', max_value= 77777777)
    st.write('Monthly Charges')
    TotalCharges = st.number_input('   ', max_value= 797777777)
    st.write("Total Charges")
    st.write(" ")
    df = pd.DataFrame({'gender': [Gender], 'Partner': [Partner], 'Dependents': [Dependents], 'PhoneService': [PhoneService], 'MultipleLines':[MultipleLines], 'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection], 'TechSupport':[TechSupport], 'StreamingTV':[StreamingTV], 'StreamingMovies':[StreamingMovies], 'Contract': [Contract], 'SeniorCitizen':[SeniorCitizen], 'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
    df = df[['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']]
    X = transformer.transform(df)
    if(st.button("Submit")):
        ans = bool((model.predict(X)[0]))
        if ans:
            st.error("Customer Is probably Going To Unsubscribe The Service")
        else:
            st.success("Customer Is Happy With The Service")

with open('techstack.html','r') as f:
  techstack=f.read();
def tech():
    components.html(
    techstack
    ,
    height=1000,
    
    scrolling=True,
    )
def ml():
  st.write("To view the complete code for the end-to-end project, visit our [GitHub](https://github.com/snshahgit/telecom-customer-churn-modelling)")
  components.iframe("https://www.kaggle.com/embed/sns5154/telecom-customer-churn-modelling-accuracy-83?kernelSessionId=100308416",height=1000,)





if choose=="Predictor":

    pred()
elif choose=="Home":
    st.title('AI for Business Development')
    st.markdown("<p style='text-align: justify;'>The objective of the project is to diagnostically predict whether or not a patient has Type 2 diabetes. \nThis predictor is built for Women above 21 years of age. The dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, used for this project consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.</p>", unsafe_allow_html=True)

    # st.markdown("<h1 style='text-align: center;'>Healthcare AI</h1>", unsafe_allow_html=True)

    with open("pic.html",'r') as f:
        pic=f.read();
    components.html(pic, height=400)

    # def load_lottieurl(url: str):
    #     r = requests.get(url)
    #     if r.status_code != 200:
    #         return None
    #     return r.json()
 
    # lt_url_hello = "https://assets6.lottiefiles.com/packages/lf20_1yy002na.json"
    # lottie_hello = load_lottieurl(lt_url_hello)
 
    # st_lottie(
    #         lottie_hello,  
    #         key="hello",
    #         speed=1,
    #         reverse=False,
    #         loop=True,
    #         quality="low",
    #         height=400,
    #         width=400            
    # )

    
elif choose=="Tech Stack":
    tech()
elif choose=="Contributors":
    html()
elif choose=="ML Code":
  ml()
