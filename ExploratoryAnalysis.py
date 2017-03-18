# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:37:49 2016

@author: lchlebak
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Turn file into a dataframe.
df = pd.DataFrame.from_csv('/home/lchlebak/Downloads/training_data.csv')

#Add new column that considers the difference between Current and Baseline
#Parameter 1.
df['dP1']=df['Current Parameter 1']-df['Baseline Parameter 1'].shift(-1)

#Create a new column that extracts only the Current Parameter 2 measurement,
#and not the associated percentage.
df['CP2']=df['Current Parameter 2'].str.extract('(\d+.\d )', expand=True)

#Add new column that considers the difference between Baseline and Current
#Parameter 2. Need to convert entries in CP2 to numerical values.
df['dP2']=df['Baseline Parameter 2']-pd.to_numeric(df['CP2'])

#Compare the difference in Parameter 1 to the difference in Parameter 2 to 
#see if there is a relationship.
sns.lmplot('dP1', 'dP2', 
            data=df, 
            fit_reg=False,  
            hue="Final Triage",
            scatter_kws={"marker": "D", 
                         "s": 10})
plt.title('Difference in Parameter 1 vs. Difference in Parameter 2')
plt.xlabel('Difference in Parameter 1')
plt.ylabel('Difference in Parameter 2')

#Subdivide data into female and male patients.
dfF = df.groupby(['Gender']).get_group('female')
dfM = df.groupby(['Gender']).get_group('male')

#Difference in Parameter 1 versus difference in Parameter 2 for different Gender
#groups, with designated Final Triage group labeled.
g = sns.lmplot(x="dP1", y="dP2", col="Gender", hue="Final Triage",
               fit_reg=False, data=df)

#Subdivide data based on Final Triage.
dfT1 = df.groupby(['Final Triage']).get_group(1)
dfT2 = df.groupby(['Final Triage']).get_group(2)
dfT3 = df.groupby(['Final Triage']).get_group(3)
dfT4 = df.groupby(['Final Triage']).get_group(4)

#Difference in Parameter 1 versus difference in Parameter 2 for different Final 
#Triage groups, with designated Gender labeled.
h = sns.lmplot(x="dP1", y="dP2", col="Final Triage", hue="Gender",
               fit_reg=False, data=df)