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
                         icons=['house', 'stack', 'cpu','terminal', 'people-fill'],
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
    st.title("TELECOM CUSTOMER CHURN MODEL")

    Gender = st.selectbox(" ", ('Male', 'Female'))
    st.write('Gender of the customer')
    Partner = st.selectbox("  ", ('Yes', 'No'))
    st.write('Marital Status ?')
    Dependents = st.selectbox("   ", ('Yes', 'No'))
    st.write('Does the customer have any dependents ?')
    PhoneService = st.selectbox("    ", ('Yes', 'No'))
    st.write('Has the customer availed the Phone Service ?')
    MultipleLines = st.selectbox("     ", ('Yes', 'No','No phone service'))
    st.write('Has the customer availed Multiple Lines ?')
    InternetService = st.selectbox("      ", ('Fiber optic','DSL', 'No'))
    st.write('Does the customer use Internet Service ?')
    OnlineSecurity = st.selectbox("        ", ('Yes', 'No','No internet service'))
    st.write('Has the customer opted for Online Security ?')
    OnlineBackup = st.selectbox("                  ", ('Yes', 'No','No internet service'))
    st.write('Has the customer opted for Online Backup ?')
    DeviceProtection = st.selectbox("                   ", ('Yes', 'No','No internet service'))
    st.write('Has the customer opted for Device Protection ?')
    TechSupport = st.selectbox("                              ", ('Yes', 'No','No internet service'))
    st.write('Has the customer opted for Tech Support ?')
    StreamingTV = st.selectbox("                                    ", ('Yes', 'No','No internet service'))
    st.write('Does the customer stream TV ?')
    StreamingMovies = st.selectbox("                                   ", ('Yes', 'No','No internet service'))
    st.write('Does the customer stream movies ?')
    Contract = st.selectbox("                                                 ", ('Month-to-month','One year','Two year'))
    st.write('What is the term of contract ?')
    SeniorCitizen = st.selectbox("                 ", ('Yes', 'No'))
    st.write('Is the customer a senior citizen ?')
    if SeniorCitizen=='Yes':
        SeniorCitizen=1
    else:
        SeniorCitizen=0
    MonthlyCharges = st.number_input(' ', max_value= 1000000)
    st.write('Monthly cost charged to the customer ?')
    TotalCharges = st.number_input('   ', max_value= 50000000)
    st.write("Total cost charged to the customer for the present term")
    st.write(" ")
    df = pd.DataFrame({'gender': [Gender], 'Partner': [Partner], 'Dependents': [Dependents], 'PhoneService': [PhoneService], 'MultipleLines':[MultipleLines], 'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection], 'TechSupport':[TechSupport], 'StreamingTV':[StreamingTV], 'StreamingMovies':[StreamingMovies], 'Contract': [Contract], 'SeniorCitizen':[SeniorCitizen], 'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
    df = df[['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']]
    X = transformer.transform(df)
    if(st.button("Submit")):
        ans = bool((model.predict(X)[0]))
        if ans:
            st.error("The customer might plan to unsubscribe the service")
        else:
            st.success("The customer seems to enjoy the service")

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
    
    st.subheader("Business Aspect")
    st.markdown("<p style='text-align: justify;'>Apart from revenue generation and positive sentiment, a crucial growth parameter for any business is <b><i>retention</i></b> of its customers. A technical term for this is called <b><i>churn</i></b>. This parameter provides insights on how strongly an existing customer is motivated to leave the service/product sold by a company. Later, special deals can be offered to the ones who have a higher churn score.</p>", unsafe_allow_html=True)
    st.write('')
    st.write('')

    st.subheader("How does this work ?")
    st.markdown("<p style='text-align: justify;'>This project diagnostically predicts whether a customer of a telecom company plans on continuing the service for next term. Customer data used to train our Machine Learning model includes data such as Gender, Marital status, different types of services, type of contract, streaming preferences and the charges incurred by the customer.</p>", unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.subheader("What next ?")
    st.markdown("<p style='text-align: justify;'>The results can be used as inputs for a dynamic pricing system which offers great deals to retain the customer. These results can also be fed to a recommendation engine which recommends plans/services which are affordable to the customer.</p>", unsafe_allow_html=True)

    # st.markdown("<h1 style='text-align: center;'>Healthcare AI</h1>", unsafe_allow_html=True)

    # with open("pic.html",'r') as f:
    #     pic=f.read();
    # components.html(pic, height=400)

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
