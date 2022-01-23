
# 
import os

from pandas.io.formats import style
from pandas.io.parsers import count_empty_vals
import streamlit as st 
import pandas as pd 
import numpy as np 
from datetime import timedelta
# 
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Get timeSeries data for a Country
def get_time_series(full_table,country):
      # for some countries data is spread over Provinces
  if full_table[full_table['Country/Region']== country]['Province/State'].nunique()>1:
    country_table = full_table[full_table['Country/Region'] == country]
    country_df = pd.DataFrame(pd.pivot_table(country_table, values=['Confirmed', 'Deaths','Recovered','Active'],
                                             index = 'Date', aggfunc=sum).to_records())
    return country_df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]
  df = full_table[(full_table['Country/Region'] == country) 
                & (full_table['Province/State'].isin(['', country]))]
  return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]

def get_time_series_province(full_table,province):
    # for some countries, data is spread over several Provinces
    df = full_table[(full_table['Province/State'] == province)]
    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]

# Download data from Github and Merge in Single Table

def load_data(country_name):
    #check the old format
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    confirmed_table = confirmed_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Confirmed").fillna('').drop(['Lat', 'Long'], axis=1)
    death_table = death_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Deaths").fillna('').drop(['Lat', 'Long'], axis=1)
    recovered_table = recovered_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Recovered").fillna('').drop(['Lat', 'Long'], axis=1)
    full_table = confirmed_table.merge(death_table).merge(recovered_table)
    full_table['Date'] = pd.to_datetime(full_table['Date'])
    # Add active cases column
    #active = Conf-deaths-recoverd
    full_table["Active"] = full_table['Confirmed']-full_table['Deaths']-full_table['Recovered']
    # replacing Mainland china with just China
    full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')
    # filling missing values
    full_table[['Province/State']] = full_table[['Province/State']].fillna('')

    ## country name search
    country = country_name
    df = get_time_series(full_table,country)
    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:
        df.drop(df.tail(1).index,inplace=True)

    df=df[df['Confirmed'] !=0]
    #st.table(df.head())
    df.to_csv("data/"+country_name+"/dataset.csv", index=True)
    return


# Linear Prediction for the next 30 days
def linear_prediction(train_ml, valid_ml, mode):
    linear_model = LinearRegression(normalize=True)
    linear_model.fit(np.array(train_ml["Days_Since"]).reshape(-1,1), np.array(train_ml[mode]).reshape(-1,1))
    con_prediction_valid_linreg = linear_model.predict(np.array(valid_ml["Days_Since"]).reshape(-1,1))
    rmse_score = np.sqrt(mean_squared_error(valid_ml[mode],con_prediction_valid_linreg))
    return linear_model, rmse_score


def main():
    st.write("# COVID-19 Dashboard")
    st.sidebar.header("Uer Selection")
    country_name = st.sidebar.selectbox("Select Country Name", ["Bangladesh","India","Pakistan"])
    if os.path.exists("data/"+country_name):
        if os.path.isfile("data/"+country_name+"/dataset.csv"):
            pass
        else:
            load_data(country_name)
    else:
        os.mkdir("data/"+country_name)
        load_data(country_name)
    
    df = pd.read_csv("data/"+country_name+"/dataset.csv")
    
    st.write("The COVID-19 Cases in {}".format(country_name))
    st.line_chart(df)
    
    num_last_days = st.sidebar.slider("Select the number of days you want to be display in the Summary Table.", 1 , 30)
    st.write("# The Last {} days COVID Summary Table For {}".format(num_last_days,country_name))
    st.write("This table includes the number of cases, deaths, new cases and moving average for your selection.")
    st.table(df.tail(num_last_days))

    # Predictions Model
    df["Days_Since"] = df.index - df.index[0]
    df["Date"] = pd.to_datetime(df['Date'], errors='coerce')
    
    model_selection = st.sidebar.selectbox("Select the Prediction model",
    ["Linear Regression","Polynomial Regression","Facebook Prophet Model"])

    # Split the dataframe
    train_ml = df.iloc[:int(df.shape[0]*0.95)]
    valid_ml = df.iloc[int(df.shape[0]*0.95):]
    linear_model, rmse_score = linear_prediction(train_ml, valid_ml, "Confirmed")
    new_date=[]
    last_index = df.shape[0]-1
    new_prediction_lr=[]
    # new_prediction_svm=[]
    for i in range(1,30):
        new_date.append(df['Date'].iloc[last_index]+timedelta(days=i))
        new_prediction_lr.append(linear_model.predict(np.array(df["Days_Since"].max()+i).reshape(-1,1))[0][0])
        # new_prediction_svm.append(svm_model.predict(np.array(df["Days_Since"].max()+i).reshape(-1,1))[0])
    
    
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    model_predictions = pd.DataFrame(zip(new_date,new_prediction_lr),
                                columns=["Dates","Linear Regression Prediction"])
    # model_predictions.head(30)
    st.write("# {} Model Prediction For COVID Confirmed cases For the next {} days\n".format(model_selection,"15"))
    temp_df = model_predictions[["Dates","Linear Regression Prediction"]]
    temp_df = temp_df.set_index("Dates")
    temp_df = temp_df['Linear Regression Prediction'].apply(np.ceil)
    st.line_chart(temp_df,use_container_width =True)


if __name__=="__main__":
    main()
