# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:09:15 2024

@author: User
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the LSTM model
#model = load_model('bitcoin_price_prediction_model.h5')
model = load_model('C:/Users/User/Documents/NorthCentralUniversity/ModelDeployment/btcdeployment/bitcoin_price_prediction_model.h5')

# Load the MinMaxScaler
scaler = MinMaxScaler()

# Streamlit app
def app():
    st.title('Bitcoin Price Prediction App')

    # Input features for prediction
    sequence_length = st.slider('Select sequence length', min_value=5, max_value=30, value=10)
    user_input = st.text_input('Enter the last {} days closing prices (comma-separated)'.format(sequence_length))

    if st.button('Predict'):
        try:
            # Convert user input to a NumPy array
            user_input_array = np.array(list(map(float, user_input.split(','))))
            user_input_array = user_input_array.reshape(1, -1)

            # Normalize the user input
            user_input_scaled = scaler.fit_transform(user_input_array)

            # Predict the next day's price
            prediction_scaled = model.predict(user_input_scaled)

            # Invert the scaling for the prediction
            prediction = scaler.inverse_transform(prediction_scaled)

            st.success('Predicted Bitcoin Price for the Next Day: ${:.2f}'.format(prediction[0, 0]))

        except ValueError:
            st.error('Invalid input. Please enter {} numerical values separated by commas.'.format(sequence_length))

# Run the Streamlit app
if __name__ == '__main__':
    app()


# streamlit run "C:\Users\User\Documents\NorthCentralUniversity\ModelDeployment\btcdeployment\app.py"
