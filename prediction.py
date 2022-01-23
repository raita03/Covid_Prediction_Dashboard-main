from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
import pandas as pd
import numpy as np
# from pyramid.arima import auto_arima
# from fbprophet import Prophet


class Prediction:
    def __init__(self, data_path, next_pred=30, split_size=0.95):
        self.data_path = data_path
        self.next_pred = next_pred
        self.df = None
        self.train_data = None
        self.valid_data = None 
        self.split_size = split_size
        self.model_scores = {}
        self.model_prediction = None
    

    def spliting_dataset(self, split_size):
        self.train_data = self.df.iloc[:int(self.df.shape[0]*self.split_size)]
        self.valid_data = self.df.iloc[:int(self.df.shape[0]*self.split_size)]
    


    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def linear_regressor_prediction(self, mode):
        if mode is "Confirmed" or mode is "Deaths":
            linear_model = LinearRegression(normalize=True)
            linear_model.fit(np.array(self.train_data["Days_Since"]).reshape(-1,1), np.array(self.train_data[mode]).reshape(-1,1))
            con_prediction_valid_linreg = linear_model.predict(np.array(self.valid_data["Days_Since"]).reshape(-1,1))
            self.model_scores["linear_model_"+mode] = (np.sqrt(mean_squared_error(self.valid_data[mode],con_prediction_valid_linreg)))
            return linear_model

    def polynomial_regressor_prediction(self, mode, degree):
        if mode is "Confirmed" or mode is "Deaths":
            poly_model =  PolynomialFeatures(degree=degree)
            train_poly = poly_model.fit_transform(np.array(self.train_data["Days_Since"]).reshape(-1,1))
            valid_poly = poly_model.fit_transform(np.array(self.valid_data["Days_Since"]).reshape(-1,1))
            y= self.train_data[mode]
            linreg = LinearRegression(normalize=True)
            linreg.fit(train_poly,y)
            prediction_poly = linreg.predict(valid_poly)
            self.model_scores["polynomial_model_"+mode] = np.sqrt(mean_squared_error(self.valid_data[mode],prediction_poly))
            return linreg

    def svm_regressor_prediction(self, mode, C=1,degree=6, kernel="poly", epsilon=0.01):
        if mode is "Confirmed" or mode is "Deaths":
            svm = SVR(C=C, kernel=kernel, degree=degree,epsilon=epsilon)
            svm.fit(np.array(self.train_data["Days_Since"]).reshape(-1,1),np.array(self.train_data[mode]).reshape(-1,1))
            prediction_valid_svm = svm.predict(np.array(self.valid_data["Days_Since"]).reshape(-1,1))
            self.model_score['svm_'+mode] = (np.sqrt(mean_squared_error(self.valid_data[mode],prediction_valid_svm)))
            return svm

    def holts_linear_regressor_prediction(self, mode):
        if mode is "Confirmed" or mode is "Deaths":
            holt_model = Holt(np.asarray(self.train[mode])).fit(smoothing_level=0.3, smoothing_slope=0.4,optimized=False)
            y_pred = self.valid_data.copy()
            y_pred["Holt"]= holt_model.forecast(len(valid[m]))
            self.model_scores["holts_linear_"+mode] = (np.sqrt(mean_squared_error(y_pred[mode],y_pred["Holt"])))
            return holt_model

    def holts_winter_regressor_prediction(self, mode):
        if mode is "Confirmed" or mode is "Deaths":
            y_pred = self.valid.copy()
            es   =   ExponentialSmoothing(np.asarray(self.train_data[mode]),seasonal_periods=9,trend='add', seasonal='mul').fit()
            y_pred["Holt's Winter Model"] = es.forecast(len(self.valid_data[mode]))
            self.model_scores["holts_winter_"+mode]  =  (np.sqrt(mean_squared_error(y_pred[mode],y_pred["Holt's Winter Model"])))
            
            return es

    # def arima_regressor_prediction(self, mode):
        if mode is "Confirmed" or mode is "Deaths":
            y_pred = self.valid_data.copy()
            model_arima = auto_arima(self.train_data[mode],
                trac = True,
                error_action='ignore',
                start_p=1,
                start_q=1,
                max_p=2,
                max_q=2,
                suppress_warnings=True,
                stepwise=False,
                seasonal=False
            )
            model_arima.fit(self.train_data[mode])
            prediction_arima = model_arima.predict(len(self.valid_data[mode]))
            y_pred["ARIMA Model Prediction"] = prediction_arima
            self.model_scores["arima_model"] = (np.sqrt(mean_squared_error(y_pred[mode],y_pred["ARIMA Model Prediction"])))
            return model_arima

    # def facebook_prophet_prediction(self, mode):
    #     prophet_c= Prophet(interval_width=0.90,weekly_seasonality=False,changepoint_range=0.9)
    #     prophet_confirmed = pd.DataFrame(zip(list(self.df.Date),list(self.df[mode])),columns=['ds','y'])
    #     prophet_c.fit(prophet_confirmed)
    #     forecast_c= prophet_c.make_future_dataframe(periods = self.next_pred-1)
    #     forecast_confirmed = forecast_c.copy()
    #     confirmed_forecast = prophet_c.predict(forecast_c)
    #     self.model_scores['facebook_prophet'] = (np.sqrt(mean_squared_error(self.df[mode],confirmed_forecast['yhat'].head(self.df.shape[0]))))
    #     return prophet_c, confirmed_forecast

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

    def initialize_dataset(self, country_name):
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

