import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from streamlit_extras.app_logo import add_logo
from datetime import datetime, time
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from streamlit_extras.dataframe_explorer import dataframe_explorer

add_logo("data/washington2.jpeg", height=100)

st.sidebar.markdown("# Business Tool")

st.markdown("# Predictions Tool")

st.subheader("Make your predictions in real time")
st.text("Upload a dataset in the exact same format as given to the consultancy firm\nand make your predictions.")
uploaded_file = st.file_uploader("Please select a .csv file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    dataframe["index"] = dataframe['dteday'].astype(str) +"-"+ dataframe["hr"].astype(str)
    dataframe = dataframe.drop('instant', axis = 1)
    dataframe = dataframe.set_index('index')
    dataframe['dteday'] = pd.to_datetime(dataframe['dteday'])
    dataframe['year'] = dataframe['dteday'].dt.year
    dataframe = dataframe.drop(columns = 'yr')
    dataframe['month'] = dataframe['dteday'].dt.month
    dataframe = dataframe.drop(columns = 'mnth')
    dataframe['sunrise'] = np.where(dataframe['month']==1, '7:30:00', None)
    dataframe['sunrise'] = np.where(dataframe['month']==2, '7:15:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==3, '6:45:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==4, '7:00:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==5, '6:00:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==6, '5:45:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==7, '5:45:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==8, '6:15:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==9, '6:30:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==10, '7:00:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==11, '7:30:00', dataframe['sunrise'])
    dataframe['sunrise'] = np.where(dataframe['month']==12, '7:00:00', dataframe['sunrise'])

    dataframe['sunset'] = np.where(dataframe['month']==1, '17:00:00', None)
    dataframe['sunset'] = np.where(dataframe['month']==2, '17:30:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==3, '18:00:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==4, '19:30:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==5, '20:00:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==6, '20:30:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==7, '20:30:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==8, '20:15:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==9, '19:30:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==10, '18:45:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==11, '18:00:00', dataframe['sunset'])
    dataframe['sunset'] = np.where(dataframe['month']==12, '16:45:00', dataframe['sunset'])

    dataframe['hr'] = dataframe['hr'].apply(lambda x: datetime.strptime(str(x), '%H').strftime('%H:%M:%S'))
    dataframe['sunrise'] = dataframe['sunrise'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').strftime('%H:%M:%S'))
    dataframe['sunset'] = dataframe['sunset'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').strftime('%H:%M:%S'))

    dataframe['sunlight'] = np.where( (dataframe['sunrise'] < dataframe['hr']) & ( dataframe['sunset'] > dataframe['hr']),1, 0)

    dataframe = dataframe.drop(['sunrise','sunset'], axis = 1)

    X = dataframe.drop(columns = ['cnt', "dteday", 'casual', 'registered', 'season', 'temp'])
    X['hr'] = X['hr'].str[:2].astype(int)

    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    X['count'] = y_pred
    X['count'] = round(np.exp2(X['count']))
    X = X.loc[:, ["hr","count"]]
    st.write(X)

    summary_table = X["count"].describe()
    summary_table = summary_table.transpose()
    st.write(summary_table)

    X_interactive = X.copy()
    X_interactive["dteday"] = dataframe["dteday"]
    X_interactive['date'] = X_interactive['dteday'].apply(lambda x: x.strftime('%Y-%m-%d'))
    X_interactive = X_interactive.drop("dteday", axis=1)
    X_interactive["date"] = pd.to_datetime(X_interactive['date'])

    interactive_data = dataframe_explorer(X_interactive)
    interactive_data = pd.DataFrame(interactive_data.groupby('date')[['count']].sum())
    st.line_chart(interactive_data, use_container_width=True)
