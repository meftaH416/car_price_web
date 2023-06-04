#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 16:31:27 2023

@author: Meftah Uddin
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import streamlit as st

def main():
    html_temp = """
    <div style="background-color: lightblue; padding:16px">
    <h2 style="color:black; text-align:center">Car Price Prediction</h2>
    </div>
    """

    model = xgb.Booster()
    model.load_model("xgb_model.json")
    

    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    st.markdown("Are you selling your car?")
    st.markdown("Let's evaluate it")
    
    p1 = st.number_input("What is the current price of the car (In Thousand)", 1.0,20.5,step=(0.2))
    p2 = st.number_input("What is the milage of the car? (In miles)")
    
    s1 = st.selectbox("What is the Fuel type",('Petrol', 'Diesel', 'CNG'))
    if s1=='Petrol':
        p3 = 0
    elif s1=='Diesel':
        p3=1
    elif s1=='CNG':
        p3=3
        
    s2 = st.selectbox("Are you dealer of individual?",('Dealer', 'Individual'))
    if s2=='Dealer':
        p4 = 0
    elif s2=='Individual':
        p4=1
        
    s3 = st.selectbox("What type of transmission?",('Manual', 'Automatic'))
    if s3=='Manual':
        p5 = 0
    elif s3=='Automatic':
        p5=1
         
    p6 = st.slider("Number of owners the car previously had",0,10)
    
    date_time = datetime.datetime.now()
    years = st.number_input("The year of the car", 1990, date_time.year)
    
    p7 = date_time.year - years
    
    data_new = pd.DataFrame({
    'Present_Price':p1,
    'Kms_Driven':p2,
    'Fuel_Type':p3,
    'Seller_Type':p4,
    'Transmission':p5,
    'Owner':p6,
    'Age':p7
   },index=[0])
    
    # Convert DataFrame to DMatrix
    dtest = xgb.DMatrix(data_new)
    
    try:
        
        if st.button('Predict'):
            pred = model.predict(dtest)
            
            if pred>0:  
                st.balloons()
                st.success("You can sell your car for {:.2f} thousand USD".format(pred[0]))
                
            else:
                st.Warning("You can't sell the car")
                
    except:
            st.Warning("Something wrong, Please try again")
if __name__ == '__main__':
    main()
