import pandas as pd
import numpy as np
import streamlit as st
import sklearn
data_file_path = "cleaned_data.csv"
columns_file_path="columns_data.csv"
df = pd.read_csv(data_file_path)

x=df.drop('price',axis='columns')
y=df.price
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(x_train,y_train)
df1 = pd.read_csv(columns_file_path)
st.title('House Price Prediction')


# Input fields for user input
sqft = st.text_input('Enter Square Footage', '1500')
bath = st.text_input('Enter Number of Bathrooms', '2')
bhk = st.text_input('Enter Number of Bedrooms (BHK)', '2')
location = st.selectbox('Location', df1['location'].unique())

def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(x_train.columns==location)[0][0]
    xn=np.zeros(len(x_train.columns))
    xn[0]=sqft
    xn[1]=bath
    xn[2]=bhk
    if loc_index>=0:
        xn[loc_index]=1
    return lr_clf.predict([xn])[0] 

if st.button('Predict Price'): 
    prediction = predict_price(location,sqft,bath,bhk)
    st.success(f'Estimated Price: {prediction:,.2f}Lakh')   
    
