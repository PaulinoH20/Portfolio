import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.app_logo import add_logo
from streamlit_extras.echo_expander import echo_expander
from datetime import datetime, time
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

add_logo("data/washington2.jpeg", height=100)

st.sidebar.markdown("# Machine Learning Process")

st.markdown("# Machine Learning")
st.text("After doing the data preprocessing, we can start working on our model.\nOur goal is to predict the total amount of users.\nSince all the users are going to need a bike.\nWe will acomplish our goal by making use of regression algorithms.")
st.text("Our main focus is to identify patterns to determine when bikes need to be available and when less so.\nHaving the appropiate amount of bikes in circulation ensures we keep running costs to a minimum,\nas having bikes not in use under bad weather conditions will deteriorate them.")

st.subheader("Feature Engineering")
st.text("In order to improve our data, we decided to engineer new features which would help\nus improve the model.")
st.text("We created the columns 'sunrise', 'sunset' and 'sunlight' in order to attempt to\nexplain how the time of the day affects the total usage of bikes, as not eveyone\nrides them during the same hours of the day, there is going to be a difference\nbetween night time hours and daylight hours.\nAfter having implemented 'sunlight', we dropped the other two columns.")
st.text("Additionally, once we obtain our predictions, we need to square the values to \nundo previous transformations and obtain the actual count.")

st.subheader("Modeling the data")
st.text("In this step we set up the algorithms for the models we will be using.")

st.text("We selected algorithms that would allow us to make numerical predictions,\nas our aim is to calculate the amount of bikes that will be needed at different\npoints in time.")

st.subheader("Model Evaluation")
st.markdown("- XGBRegressor: works by iteratively adding decision trees to the model, each tree attempting to correct the errors made by the previous trees. The algorithm uses a gradient-based approach to minimize the mean squared error (MSE) between the predicted values and the actual values of the target variable.")
st.markdown("- LGBMRegressor: is a model for regression tasks, based on the gradient boosting framework. It is an implementation of the LightGBM algorithm, which is a high-performance, distributed gradient boosting framework that uses tree-based learning algorithms.")
st.markdown("- K-neighbours: is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.")
st.markdown("- Random Forest Regressor: is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.")
st.markdown("- Linear Regression: is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables.")
st.text("Now we compare the score of the metrics we used to evaluate the model\nand choose the one that suits our needs best.")

modelScores = {
    'Model': ['Linear Regression', "K-neighbours", "Random Forest Regressor", "LGBMRegressor", "XGBRegressor"],
    'Mean Absolute Error': [0.97, 0.38, 0.53, 0.31, 0.30],
    'Mean Squared Error': [1.64, 0.31, 0.51, 0.22, 0.21],
    'Root Mean Squared Error': [1.28, 0.56, 0.71, 0.47, 0.46],
    'R2 Score': [0.64, 0.93, 0.89, 0.95, 0.95]
}

modelScores = pd.DataFrame(modelScores)
modelScores = modelScores.sort_values(by='Root Mean Squared Error', ascending=True)
st.write(modelScores)

st.subheader("Deploying Model")
st.text("Once we have our best mode, on this case being XGBRegressor, because it returned\nthe best metrics for our business solution")
st.text("Now that we have our trained model we need to save it for future use, so that our\nclients can make predictions.")