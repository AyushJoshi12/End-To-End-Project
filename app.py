import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# load the dataset
cal = fetch_california_housing()
df = pd.DataFrame(data=cal.data, columns=cal.feature_names)
df['price'] = cal.target
df.head()

# title of the app
st.title("California Housing Prices App")

# data overview

st.subheader("Data Overview")
st.dataframe(df.head(10))

# split the data into train and test
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the data
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# model selection
st.subheader("Select A Model")

model = st.selectbox("Choose a model", ['Model 1 : Linear Regression', 'Model 2 : Ridge', 'Model 3 : Lasso', 'Model 4 : ElasticNet'])

# initialize the model
models = {'Model 1 : Linear Regression' : LinearRegression(), 
          'Model 2 : Ridge' : Ridge(), 
          'Model 3 : Lasso' : Lasso(), 
          'Model 4 : ElasticNet' : ElasticNet()}

# train the selected model
selected_model = models[model]

selected_model.fit(X_train_sc, y_train)

# predict the model
y_pred = selected_model.predict(X_test_sc)

# evaluate the model
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2_score = r2_score(y_test, y_pred)

# display the metrics for selected model
st.write("Test MSE", test_mse)
st.write("Test MAE", test_mae)
st.write("Test RMSE", test_rmse)
st.write("Test R2 Score", test_r2_score)

# prompt the user to enter the input values
st.write("Enter the input values to predict house price: ")

user_input = {}

for feature in X.columns:
    user_input[feature] = st.number_input(feature)

user_input_df = pd.DataFrame([user_input])

# scale the user input
user_input_sc = sc.transform(user_input_df)

# predict the house price
predicted_price = selected_model.predict(user_input_df)

# display the predicted house price
st.write(f"Predicted House Price: {predicted_price[0]*100000}")