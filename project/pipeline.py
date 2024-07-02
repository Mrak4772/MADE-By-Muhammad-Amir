import numpy as np
import pandas as pd
import sqlite3
import os
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import datetime as dt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths to the datasets
datasets = {
    "CO2": './data/archive.csv',
    "city_temp": './data/ClimateChange/GlobalLandTemperaturesByCity.csv',
    "country_temp": './data/ClimateChange/TemperaturesByCountry.csv',
    "major_city_temp": './data/ClimateChange/TemperaturesByMajorCity.csv',
    "state_temp": './data/ClimateChange/TemperaturesByState.csv',
    "global_temp": './data/ClimateChange/GlobalTemperatures.csv'
}

# Load datasets
CO2_df = pd.read_csv(datasets["CO2"])
temp_by_city = pd.read_csv(datasets["city_temp"])
temp_by_country = pd.read_csv(datasets["country_temp"])
temp_by_major_city = pd.read_csv(datasets["major_city_temp"])
temp_by_state = pd.read_csv(datasets["state_temp"])
global_temp = pd.read_csv(datasets["global_temp"])

# Drop NaN values
CO2_df.dropna(inplace=True)
temp_by_city.dropna(inplace=True)
temp_by_country.dropna(inplace=True)
temp_by_major_city.dropna(inplace=True)
temp_by_state.dropna(inplace=True)
global_temp.dropna(inplace=True)

# Modify the date format to extract year
def to_year(date):
    return int(date.split('-')[0])

# Apply date modification and create a new column called 'year'
temp_by_state['year'] = temp_by_state['dt'].apply(to_year)
temp_by_country['year'] = temp_by_country['dt'].apply(to_year)
temp_by_major_city['year'] = temp_by_major_city['dt'].apply(to_year)
global_temp['year'] = global_temp['dt'].apply(to_year)

# Ensure the data directory exists
os.makedirs('/data', exist_ok=True)

# Function to save DataFrame to SQLite
def save_to_sqlite(df, db_name, table_name):
    conn = sqlite3.connect(f'/data/{db_name}.db')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# Save each dataset to SQLite
save_to_sqlite(CO2_df, 'climate_data', 'CO2')
save_to_sqlite(temp_by_city, 'climate_data', 'temp_by_city')
save_to_sqlite(temp_by_country, 'climate_data', 'temp_by_country')
save_to_sqlite(temp_by_major_city, 'climate_data', 'temp_by_major_city')
save_to_sqlite(temp_by_state, 'climate_data', 'temp_by_state')
save_to_sqlite(global_temp, 'climate_data', 'global_temp')

print("Datasets have been successfully processed and saved to the /data directory.")

# Further analysis and visualizations
# Collecting the average temperature per year
dfa = pd.DataFrame()
years = temp_by_country['year'].unique()
for i in years:
    df_avg = temp_by_country[temp_by_country['year'] == i]['AverageTemperature'].mean()
    df_new = (temp_by_country[temp_by_country['year'] == i]).head(1)
    df_new['AverageTemperature'] = df_avg
    dfa = pd.concat([dfa, df_new], ignore_index=True)

# Average Temperature above 9 degrees
df_above_nine = dfa[dfa['AverageTemperature'] >= 9]
df_above_nine.plot.scatter(x='year', y='AverageTemperature', c='AverageTemperature', cmap='coolwarm')
plt.title('Average Temperature above 9 degrees')
plt.show()

# Average Temperature below 9 degrees
df_below_nine = dfa[dfa['AverageTemperature'] < 9]
df_below_nine.plot.scatter(x='year', y='AverageTemperature', c='AverageTemperature', cmap='coolwarm')
plt.title('Average Temperature below 9 degrees')
plt.show()

# Processing CO2 dataset
dfc = pd.DataFrame()
years = CO2_df['Year'].unique()
for i in years:
    df_avg = CO2_df[CO2_df['Year'] == i]['Carbon Dioxide (ppm)'].mean()
    df_new = CO2_df[CO2_df['Year'] == i].head(1).copy()
    df_new['Carbon Dioxide (ppm)'] = df_avg
    dfc = pd.concat([dfc, df_new], ignore_index=True)

# Changing the Year column to year (lowercase)
dfc.rename(columns={"Year": "year"}, inplace=True)

# Dropping all unwanted columns
dfc.drop(['Seasonally Adjusted CO2 (ppm)', 
           'Carbon Dioxide Fit (ppm)', 
           'Seasonally Adjusted CO2 Fit (ppm)',
          'Decimal Date',
          'Month'], inplace=True, axis=1)

dfc = dfc.dropna()
sns.lmplot(x='year', y='Carbon Dioxide (ppm)', data=dfc)
plt.title('CO2 Levels Over Time')
plt.show()

# Heatmap of correlations
sns.heatmap(dfc.corr(), annot=True)
plt.title('Correlation Matrix of CO2 Data')
plt.show()

# Merge average temperature data with CO2 data
dfsc = pd.merge(dfa, dfc, on='year').dropna()
dfsc.drop(['Seasonally Adjusted CO2 (ppm)', 
           'Carbon Dioxide Fit (ppm)', 
           'Seasonally Adjusted CO2 Fit (ppm)',
          'Decimal Date',
          'Month'], inplace=True, axis=1)

sns.lmplot(x='AverageTemperature', y='Carbon Dioxide (ppm)', data=dfsc)
plt.title('Average Temperature vs. CO2 Levels')
plt.show()

# Plotly visualizations
grp1 = CO2_df.groupby(["year"]).mean()["Carbon Dioxide (ppm)"]
trace1 = go.Bar(x=grp1.index, y=grp1.values)
layout = go.Layout(
    title="Average CO<sub>2</sub> Levels in Atmosphere per Year",
    yaxis=dict(title="Parts per million (PPM)", range=(300,420)),
    xaxis=dict(title="Year"))
figure = go.Figure(data=[trace1], layout=layout)
py.iplot(figure, filename="co2-ppm-year")

group2 = CO2_df.groupby(["year", "Month"]).mean()["Carbon Dioxide (ppm)"]
x = [dt.datetime(year=i[0], month=i[1], day=15) for i in group2.index]

# Mean values.
y1 = group2.values

# Rolling window average
y2 = group2.rolling(3, min_periods=1).mean().values

# Exponentially weighted moving average
y3 = group2.ewm(span=3, min_periods=1).mean().values
second_trace = go.Scatter(x=x, y=y1, mode="markers", name="Actual value")
third_trace = go.Scatter(x=x, y=y2, line=dict(color="red"), name="Rolling average")
forth_trace = go.Scatter(x=x, y=y3, line=dict(color="green"), name="EWM average")

default_period = (dt.datetime(2008, 1, 1), dt.datetime(2017,12,1))
default_ppm_range = (380, 410)
layout = go.Layout(
    title="Seasonal Fluctuations of CO<sub>2</sub> Levels in Atmosphere",
    yaxis=dict(title="Parts per million (PPM)", range=default_ppm_range),
    xaxis=dict(title="Year", range=default_period))

figure = go.Figure(data=[second_trace, third_trace, forth_trace], layout=layout)
py.iplot(figure, filename="co2-ppm-seasonal")

# Linear regression model
x_val = [(i.year, i.month, i.month ** 2, i.year ** 2) for i in x]
y_val = [i for i in y1]

x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.40, random_state=45)
linearModel = linear_model.LinearRegression().fit(x_train, y_train)
print("Accuracy: ", linearModel.score(x_test, y_test))

# Predicted values
pred_value = linearModel.predict(x_val)

# Define timeline of years for prediction
predicted_years = range(1950, 2055)
predicted_months = range(1, 13)

predicted_x = []
for y in predicted_years:
    for j in predicted_months:
        predicted_x.append([y, j, j ** 2, y ** 2])
        
# Predict values
predicted_y = linearModel.predict(predicted_x)

x_plot = [dt.datetime(i[0], i[1], 15) for i in predicted_x]
fifth_trace = go.Scatter(x=x_plot, y=predicted_y, line=dict(color="red"), name="Predicted value")

period_default = (dt.datetime(1956, 1, 1), dt.datetime(2050,12,1))
ppm_range_def = (300, 500)
layout = go.Layout(
    title="Predicted Vs. Actual CO<sub>2</sub> Concentration Levels",
    yaxis=dict(title="Parts per million (PPM)", range=ppm_range_def),
    xaxis=dict(title="Year", range=period_default))
figure = go.Figure(data=[second_trace, fifth_trace], layout=layout)
py.iplot(figure, filename="co2-ppm-prediction")

# State that had the highest average temperature level
max_temp_state = temp_by_state.loc[temp_by_state['AverageTemperature'].idxmax()]
print("State with the highest average temperature level:", max_temp_state)

# Country that had the highest average temperature
max_temp_country = temp_by_country.loc[temp_by_country['AverageTemperature'].idxmax()]
print("Country with the highest average temperature level:", max_temp_country)

# Record with the highest temperature uncertainty
max_temp_uncertainty = temp_by_state.loc[temp_by_state['AverageTemperatureUncertainty'].idxmax()]
print("Record with the highest temperature uncertainty:", max_temp_uncertainty)

# Merge country temperature data with CO2 data
def mod_year(date):
    return int(date.split('-')[0])

def mod_month(date):
    return int(date.split('-')[1])

temp_by_country['year'] = temp_by_country['dt'].apply(mod_year)
temp_by_country['month'] = temp_by_country['dt'].apply(mod_month)
country_new_temp_data = pd.merge(temp_by_country, CO2_df, on=['year'])

# Ensure the 'Carbon Dioxide (ppm)' column is numeric
country_new_temp_data["Carbon Dioxide (ppm)"] = pd.to_numeric(country_new_temp_data["Carbon Dioxide (ppm)"], errors="coerce")

# Drop rows with NaN values in the 'Carbon Dioxide (ppm)' column
country_new_temp_data = country_new_temp_data.dropna(subset=["Carbon Dioxide (ppm)"])

# Select only numeric columns
numeric_data = country_new_temp_data.select_dtypes(include=[float, int])

# Include the 'year' and 'month' columns for grouping
numeric_data["year"] = country_new_temp_data["year"]
numeric_data["month"] = country_new_temp_data["month"]

# Group by year and month, then compute the mean
country_carbon = numeric_data.groupby(["year", "month"]).mean()["Carbon Dioxide (ppm)"]

# Create datetime objects for plotting
xx = [dt.datetime(year=i[0], month=i[1], day=15) for i in country_carbon.index]

# Mean values
yy1 = country_carbon.values

# Rolling window average
yy2 = country_carbon.rolling(3, min_periods=1).mean().values

# Exponentially weighted moving average
yy3 = country_carbon.ewm(span=3, min_periods=1).mean().values

# Create traces for plotting
second_country_trace = go.Scatter(x=xx, y=yy1, mode="markers", name="Actual value")
third_country_trace = go.Scatter(x=xx, y=yy2, line=dict(color="red"), name="Rolling average")
forth_country_trace = go.Scatter(x=xx, y=yy3, line=dict(color="green"), name="EWM average")

# Combine traces into a figure
fig = go.Figure(data=[second_country_trace, third_country_trace, forth_country_trace])
fig.show()

# Linear regression model for country data
x_values = [(i.year, i.month, i.month ** 2, i.year ** 2) for i in xx]
y_values = [i for i in yy1]

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.40, random_state=45)
linearModel = linear_model.LinearRegression().fit(x_train, y_train)
print("Accuracy: ", linearModel.score(x_test, y_test))

# Predicted values
pred_value = linearModel.predict(x_values)
new_predicted_x = []
for y in predicted_years:
    for j in predicted_months:
        new_predicted_x.append([y, j, j ** 2, y ** 2])

# Predict values
new_predicted_y = linearModel.predict(new_predicted_x)

new_x_plot = [dt.datetime(i[0], i[1], 15) for i in new_predicted_x]
fifth_new_trace = go.Scatter(x=new_x_plot, y=new_predicted_y, line=dict(color="red"), name="Predicted value")

layout = go.Layout(
    title="Predicted Vs. Actual CO<sub>2</sub> Concentration Levels",
    yaxis=dict(title="Parts per million (PPM)", range=ppm_range_def),
    xaxis=dict(title="Year", range=period_default))
figure = go.Figure(data=[second_country_trace, fifth_new_trace], layout=layout)
py.iplot(figure, filename="co2-ppm-prediction")
