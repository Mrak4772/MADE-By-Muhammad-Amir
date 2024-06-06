#!/usr/bin/env python
# coding: utf-8

# ### Example Report: Analysis of Data Scientists Salaries
# 
# This example uses open data from Kaggel (https://www.kaggle.com/datasets/vladimirmijatovic/data-scientists-salaries-worldwide-annual-survey)
# 
# Every year there is an annual thread on Salaries in Data Science on Reddit. Data scientists from all over the world self-report their own salaries, bonuses and other compensation. Some of the salaries are insanely high!
# 
# ## Load data
# Create a pandas dataframe using the local sqlite file

# In[210]:


import numpy as np # type: ignore # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[211]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re 


# In[212]:


df = pd.read_csv("../data/data_scientists_salaries_data.csv")


# In[213]:


df.shape


# In[214]:


df.columns


# ### Calculating Missing Values

# In[215]:


# Calculate missing values
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Combine into a DataFrame
missing_data = pd.DataFrame({
    'Total Missing': missing_values,
    'Percentage Missing': missing_percentage
}).sort_values(by='Total Missing', ascending=False)

print(missing_data)


# ## Data Cleaning

# In[217]:


# Function to clean and convert salary
def convert_salary(salary):
    if isinstance(salary, str):
        # Remove any non-numeric characters except for decimal points and 'k'
        salary = re.sub(r'[^\d\.k]', '', salary)
        # Handle 'k' (thousands) suffix
        if 'k' in salary:
            salary = salary.replace('k', '')
            try:
                return float(salary) * 1000
            except ValueError:
                return None
        try:
            return float(salary)
        except ValueError:
            return None
    return salary


# In[228]:


# Impute missing values
# Remove commas from 'salary' and convert to numeric
# Function to convert salary to numeric

df['salary'] = df['salary'].apply(convert_salary)
df['total_comp'] = df['total_comp'].apply(convert_salary)

df['salary'].fillna(df['salary'].median(), inplace=True)
df['bonus'].fillna(0, inplace=True)
df['stocks'].fillna(0, inplace=True)
df['total_comp'].fillna(df['total_comp'].median(), inplace=True)
df['tenure_length_period'].fillna(df['tenure_length_period'].median(), inplace=True)

df['title'].fillna(df['title'].mode()[0], inplace=True)
df['location'].fillna(df['location'].mode()[0], inplace=True)
df['company_industry'].fillna(df['company_industry'].mode()[0], inplace=True)
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['prior_experience'].fillna('Unknown', inplace=True)

# Normalize salary and compensation data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['salary', 'bonus', 'stocks', 'total_comp']] = scaler.fit_transform(df[['salary', 'bonus', 'stocks', 'total_comp']])

# Handle outliers by capping at the 99th percentile
cap_at_99th = lambda x: x.clip(upper=x.quantile(0.99))

df['salary'] = cap_at_99th(df['salary'])
df['total_comp'] = cap_at_99th(df['total_comp'])

df.head()
print(df)


# In[ ]:





# In[219]:


df.head()


# In[229]:


# Calculate missing values
missing_values_in_Cleandf = df.isnull().sum()
missing_percentage_in_Cleandf = (df.isnull().sum() / len(df)) * 100

# Combine into a DataFrame
missing_data_in_cleandf = pd.DataFrame({
    'Total Missing': missing_values_in_Cleandf,
    'Percentage Missing': missing_percentage_in_Cleandf
}).sort_values(by='Total Missing', ascending=False)

print(missing_data_in_cleandf)


# In[230]:


df = df


# Creating histograms of cleaned data

# In[232]:


import matplotlib.pyplot as plt

# Clean and transform salary, bonus, stocks, and total_comp columns to numeric values

def clean_currency(value):
    if isinstance(value, str):
        value = value.replace('$', '').replace(',', '').replace('k', '000').replace('K', '000').strip()
        try:
            return float(value)
        except ValueError:
            return None
    return value

df['salary'] = df['salary'].apply(clean_currency)
df['bonus'] = df['bonus'].apply(clean_currency)
df['stocks'] = df['stocks'].apply(clean_currency)
df['total_comp'] = df['total_comp'].apply(clean_currency)
# Increase figure size
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(18, 28))

# Plotting histograms for numeric columns
df['salary'].dropna().hist(ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Salary Distribution')
axes[0, 0].set_xlabel('Salary')
axes[0, 0].set_ylabel('Frequency')

df['bonus'].dropna().hist(ax=axes[0, 1], bins=30)
axes[0, 1].set_title('Bonus Distribution')
axes[0, 1].set_xlabel('Bonus')
axes[0, 1].set_ylabel('Frequency')

df['stocks'].dropna().hist(ax=axes[1, 0], bins=30)
axes[1, 0].set_title('Stocks Distribution')
axes[1, 0].set_xlabel('Stocks')
axes[1, 0].set_ylabel('Frequency')

df['total_comp'].dropna().hist(ax=axes[1, 1], bins=30)
axes[1, 1].set_title('Total Compensation Distribution')
axes[1, 1].set_xlabel('Total Compensation')
axes[1, 1].set_ylabel('Frequency')

df['tenure_length_period'].dropna().hist(ax=axes[2, 0], bins=30)
axes[2, 0].set_title('Tenure Length Period Distribution')
axes[2, 0].set_xlabel('Tenure Length (Period)')
axes[2, 0].set_ylabel('Frequency')

df['survey_year'].hist(ax=axes[2, 1], bins=30)
axes[2, 1].set_title('Survey Year Distribution')
axes[2, 1].set_xlabel('Survey Year')
axes[2, 1].set_ylabel('Frequency')

# Plotting bar plots for categorical columns
df['title'].value_counts().head(10).plot(kind='bar', ax=axes[3, 0])
axes[3, 0].set_title('Top 10 Job Titles')
axes[3, 0].set_xlabel('Job Title')
axes[3, 0].set_ylabel('Count')

df['location'].value_counts().head(10).plot(kind='bar', ax=axes[3, 1])
axes[3, 1].set_title('Top 10 Locations')
axes[3, 1].set_xlabel('Location')
axes[3, 1].set_ylabel('Count')

df['company_industry'].value_counts().head(10).plot(kind='bar', ax=axes[4, 0])
axes[4, 0].set_title('Top 10 Company Industries')
axes[4, 0].set_xlabel('Company Industry')
axes[4, 0].set_ylabel('Count')

df['education'].value_counts().head(10).plot(kind='bar', ax=axes[4, 1])
axes[4, 1].set_title('Top 10 Education Levels')
axes[4, 1].set_xlabel('Education')
axes[4, 1].set_ylabel('Count')

df['prior_experience'].value_counts().head(10).plot(kind='bar', ax=axes[5, 0])
axes[5, 0].set_title('Top 10 Prior Experiences')
axes[5, 0].set_xlabel('Prior Experience')
axes[5, 0].set_ylabel('Count')

df['additional_benefits'].value_counts().plot(kind='bar', ax=axes[5, 1])
axes[5, 1].set_title('Additional Benefits Distribution')
axes[5, 1].set_xlabel('Additional Benefits')
axes[5, 1].set_ylabel('Count')

df['tenure_length_period_units'].value_counts().plot(kind='bar', ax=axes[6, 0])
axes[6, 0].set_title('Tenure Length Period Units Distribution')
axes[6, 0].set_xlabel('Tenure Length Period Units')
axes[6, 0].set_ylabel('Count')

# Remove the empty subplot
fig.delaxes(axes[6, 1])

# Adjust subplot parameters
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.3)

plt.tight_layout()
plt.show()


# Histogram indicates The salary, bonus, stocks, and total compensation distributions are right-skewed, indicating that most data scientists earn lower amounts, with fewer receiving high compensation. Tenure length periods are varied, suggesting common employment durations. Recent survey years reflect current trends, and 'Data Scientist' is the predominant job title. Key hubs for data science jobs are highlighted in location data. The technology sector is the leading industry, and many data scientists hold advanced degrees. Diverse prior experiences are common pathways into data science. Companies offer a range of additional benefits, and different units measure tenure length, reflecting varied standards across industries.

# In[233]:


# Define the final file path
final_file_path = '../data/cleaned_data_scientists_salaries_data.csv'

# Save the cleaned DataFrame to a CSV file in the /data directory
df.to_csv(final_file_path, index=False)

final_file_path


# In[234]:


df.head()


# In[235]:


df[['tenure_length_period', 'salary']].corr()


# In[236]:


import matplotlib.pyplot as plt
import seaborn as sns
correlation = df[['tenure_length_period', 'salary']].corr().iloc[0, 1]
print(f'Correlation between Tenure Length Period and Salary: {correlation}')

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['tenure_length_period'], df['salary'], alpha=0.5)
plt.title(f'Scatter Plot of Tenure Length Period vs Salary\n(Correlation: {correlation:.2f})')
plt.xlabel('Tenure Length Period')
plt.ylabel('Salary')
plt.grid(True)
plt.show()


# The plot shows that there is no significant linear relationship between tenure length period and salary. The correlation is weak and slightly negative, indicating that other variables likely have a more substantial impact on salary levels for data scientists.
# 
# This kind of analysis is crucial because it helps identify which factors truly influence salary, guiding more targeted investigations and decisions.

# In[ ]:





# In[ ]:




