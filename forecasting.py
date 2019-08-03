import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

def sarimax_forecast(data, periods = 12, grid_search_parameters = 2):

    parameters = pd.DataFrame()
    param_list = []
    param_seasonal_list = []
    rmse_list = []

    p = d = q = range(0, grid_search_parameters)
    assert grid_search_parameters > 1
    pdq = list(itertools.product(p, d, q)) #defines the list of all possible combinations
    seasonal_pdq = [(x[0], x[1], x[2], periods) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(data[:-periods], order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                forecast_grid = results.forecast(steps = periods)
                rmse = np.sqrt(mean_squared_error(data[-periods:], forecast_grid))

                param_list.append(param)
                param_seasonal_list.append(param_seasonal)
                rmse_list.append(rmse)

            except:
                continue

    parameters['Parameters'] = param_list
    parameters['Seaonsal Parameters'] = param_seasonal_list
    parameters['Root MSE'] = rmse_list

    selected_param = parameters.sort_values(by = 'Root MSE').iloc[0, 0]
    selected_seasonal_param = parameters.sort_values(by = 'Root MSE').iloc[0, 1]

    prediction_model = SARIMAX(data, order = selected_param, seasonal_order = selected_seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
    prediction_model_fit = prediction_model.fit()
    future_forecast = prediction_model_fit.forecast(steps = periods)

    return future_forecast

def forecast(df, steps = 12):
    test0 = pd.Series(adfuller(df['Rolling Total'].dropna(), autolag='AIC')[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    test1 = pd.Series(adfuller(df['Rolling Total'].diff().dropna(), autolag='AIC')[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    test2 = pd.Series(adfuller(df['Rolling Total'].pct_change().dropna(), autolag='AIC')[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    if test0[1] < 0.05:
        diff_forecast = sarimax_forecast(df['Rolling Total'], periods = steps)
        pct_forecast = pd.Series(np.nan, index = diff_forecast.index)

    elif test1[1] or test2[1] <= 0.05:
        df_forecast1 = sarimax_forecast(df['Rolling Total'].diff(), periods = steps)
        diff_forecast = df_forecast1.cumsum()+df['Rolling Total'].iloc[-1]

        df_forecast2 = sarimax_forecast(df['Rolling Total'].pct_change(), periods = steps)
        pct_forecast = (1+df_forecast2).cumprod()*df['Current Year Total'].iloc[-1]

    elif test1[1] and test2[1] > 0.05:
        df_forecast1 = sarimax_forecast(df['Monthly Total'].diff(), periods = steps)
        diff_forecast = df_forecast1.cumsum()+df['Monthly Total'].iloc[-1]

        df_forecast2 = sarimax_forecast(df['Monthly Total'].pct_change(), periods = steps)
        pct_forecast = (df['Monthly Total'].iloc[-1]*(1+df_forecast2).cumprod()).cumsum()
    else:
        df['Rolling Change Diff'] = df['Rolling Total'].diff()
        months_diff = df.groupby('Month')['Rolling Change Diff'].mean()
        months_list_diff = list(months_diff.values)*11
        df['Average Rolling Change Diff'] = df['Rolling Change Diff']/months_list_diff

        df_forecast1 = sarimax_forecast(df['Rolling Average Change Diff'], periods = steps)
        diff_forecast = months_list_diff[:12]*df_forecast1.cumsum()+df['Rolling Total'].iloc[-1]

        df['Rolling Change %'] = df['Rolling Total'].pct_change()
        months_pct = df.groupby('Month')['Rolling Change %'].mean()
        months_list_pct = list(months_diff.values)*11
        df['Average Rolling Change %'] = df['Rolling Change %']/months_list_pct

        df_forecast2 = sarimax_forecast(df['Average Rolling Change %'], periods = steps)
        pct_forecast = (df['Monthly Total'].iloc[-1]*(1+df_forecast2*months_list_pct[:12]).cumprod()).cumsum()

    category = pd.Series(df['Category'].iloc[-1], index = diff_forecast.index)
    forecast_df = pd.concat([category, diff_forecast, pct_forecast], axis =1)
    forecast_df.columns = ['Category', 'Difference Method', 'Percentage Method']

    return forecast_df
