
# Created by Mazharul Islam Leon
import os
import time
from pandas.io.formats import style
from pandas.io.parsers import count_empty_vals
import streamlit as st 
import pandas as pd 
import numpy as np 
from datetime import timedelta
from datetime import datetime
import altair as alt
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from prediction import Prediction

if not os.path.exists("data"):
    os.mkdir("data")





# Get timeSeries data for a Country


# Download data from Github and Merge in Single Table



def animated_alt_plot(df):
    df.reset_index(inplace=True)
    df['Dates'] = df.Dates.dt.date
    col_names =  df.columns
    handle = st.area_chart(df)
    df2 = df.iloc[10:]
    df = pd.concat([df, df2])
    handle.area_chart(df)


def alt_ploting(df):
    df.reset_index(inplace=True)
    df['Dates'] = df.Dates.dt.date
    col_names =  df.columns
    # print(col_names)
    c = alt.Chart(df).mark_area(
            line={'color':'darkgreen'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0),
                    alt.GradientStop(color='darkred', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            alt.X('Dates:T'),
            alt.Y('LR_Prediction:Q')
        )
    st.altair_chart(c, use_container_width=True)

    

def showing_ploting_data(df):
    df.reset_index(inplace=True)
    df['Dates'] = df.Dates.dt.date
    X = df.Dates.to_list()
    Y = df["LR_Prediction"].to_list()
    fig = plt.figure()
    plt.style.use('seaborn-darkgrid')
    plt.plot(X,Y)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Dates")
    plt.ylabel("# Cases")
    st.pyplot(fig)


def showing_data_table(df):
    df.reset_index(inplace=True)
    df['Dates'] = df.Dates.dt.date
    col_name = df.columns
    st.table(df[col_name])



# --------------- Main ---------------------

def main():
    st.write("# COVID-19 Dashboard")
    st.sidebar.header("Uer Selection")
    country_name = st.sidebar.selectbox("Select Country Name", ["Bangladesh","India","Pakistan"])
    data_path = "data/"+country_name+"/dataset.csv"
    if os.path.exists("data/"+country_name+"/dataset.csv"):
        main_prediction = Prediction(data_path)
        # death_pred = Prediction(data_path)
    else:
        os.mkdir("data/"+country_name)
        main_prediction = Prediction(data_path)
        # death_pred = Prediction(data_path)
    
    
    
    main_df = main_prediction.load_data()
    # death_df = death_pred.load_data()
    
    st.write("The Current scenario of the COVID-19 Cases in {}. The confirmed cases, deaath cases, active cases, and recoved cases are shown below figure".format(country_name))
    st.line_chart(main_df)
    
    num_last_days = st.sidebar.slider("Select the number of days you want to be display in the Summary Table.", 1 , 30)
    st.write("# The Last {} days COVID Summary Table For {}".format(num_last_days,country_name))
    st.write("This table includes the number of cases, deaths, new cases and moving average for your selection.")
    st.table(main_df.tail(num_last_days))

    # ## How many days want to next prediction
    next_pred_show = st.sidebar.slider("Select the number of days you want to see the next prediction.",1,30)
    main_prediction.next_pred = next_pred_show



    # # Predictions Model
    main_prediction.df["Days_Since"] = main_df.index - main_df.index[0]
    main_prediction.df["Date"] = pd.to_datetime(main_df['Date'], errors='coerce')
    main_prediction.spliting_dataset(split_size=0.95)
    
    model_selection = st.sidebar.selectbox("Select the Prediction model",
    ["Linear Regression",
    "Polynomial Regression",
    "Support Vectore Regressor",
    "Holts Linear Regressor",
    "Holts Winter Regressor",
    "Arima Model",
    "Facebook Prophet Model"])

    new_date=[]
    last_index = main_prediction.df.shape[0]-1
    new_prediction_lr=[]
    new_prediction_poly = []
    new_prediction_svm = []

    st.write("# {} Model Selected".format(model_selection))

    if model_selection == "Linear Regression":
        mode = "Confirmed"
        linear_model = main_prediction.linear_regressor_prediction(mode="Confirmed")
        st.write("RMSE Score by {}: {:.2f} for predicting {} cases".format(model_selection,
        main_prediction.model_scores["linear_model_Confirmed"], mode))
        for i in range(1,next_pred_show):
                new_date.append(main_prediction.df['Date'].iloc[last_index]+timedelta(days=i))
                new_prediction_lr.append(linear_model.predict(np.array(main_prediction.df["Days_Since"].max()+i).reshape(-1,1))[0][0])
        
        model_predictions = pd.DataFrame(zip(new_date,new_prediction_lr),
                                columns=["Dates","LR_Prediction"])
        model_predictions = model_predictions.set_index("Dates")
        model_predictions = model_predictions['LR_Prediction'].apply(np.ceil)
        if st.checkbox("Shows Table"):
            showing_data_table(model_predictions[:next_pred_show].to_frame())
        
        if st.checkbox("Shows Graph"):
            alt_ploting(model_predictions[:next_pred_show].to_frame())
    elif model_selection == "Polynomial Regression":
        mode = "Confirmed"
        poly_model = main_prediction.polynomial_regressor_prediction(mode, degree=5)
        st.write("RMSE Score by {}: {:.2f} for predicting {} cases".format(model_selection,
        main_prediction.model_scores["polynomial_model_Confirmed"], mode))

        for i in range(1,next_pred_show):
            pass
        
        model_predictions = pd.DataFrame(zip(new_date,new_prediction_poly),
                                columns=["Dates","Poly_Prediction"])
        model_predictions = model_predictions.set_index("Dates")
        model_predictions = model_predictions['Poly_Prediction'].apply(np.ceil)
        if st.checkbox("Shows Table"):
            showing_data_table(model_predictions[:next_pred_show].to_frame())
        
        if st.checkbox("Shows Graph"):
            alt_ploting(model_predictions[:next_pred_show].to_frame())
    
    
    
    
    
    
    # train_ml = df.iloc[:int(df.shape[0]*0.95)]
    # valid_ml = df.iloc[int(df.shape[0]*0.95):]
    # linear_model, rmse_score = linear_prediction(train_ml, valid_ml, "Confirmed")
    # new_date=[]
    # last_index = df.shape[0]-1
    # new_prediction_lr=[]
    # # new_prediction_svm=[]
    # for i in range(1,30):
    #     new_date.append(df['Date'].iloc[last_index]+timedelta(days=i))
    #     new_prediction_lr.append(linear_model.predict(np.array(df["Days_Since"].max()+i).reshape(-1,1))[0][0])
    #     # new_prediction_svm.append(svm_model.predict(np.array(df["Days_Since"].max()+i).reshape(-1,1))[0])
    
    
    # pd.set_option('display.float_format', lambda x: '%.1f' % x)
    # model_predictions = pd.DataFrame(zip(new_date,new_prediction_lr),
    #                             columns=["Dates","LR_Prediction"])
    # # model_predictions.head(30)
    # st.write("# {} Model Prediction".format(model_selection))
    # st.write("Next {} days prediction using {} model. If you want show the table then press the show table button".format(next_pred_show,model_selection))
    # temp_df = model_predictions[["Dates","LR_Prediction"]]
    # temp_df = temp_df.set_index("Dates")
    # temp_df = temp_df['LR_Prediction'].apply(np.ceil)

    # alt_ploting(temp_df[:next_pred_show].to_frame())

    
    # # st.area_chart(temp_df,use_container_width =True)

    # ## button for showing the data in a table 
    # if st.checkbox("Show in table"):
    #     showing_data_table(temp_df[:next_pred_show].to_frame())




if __name__=="__main__":
    main()
