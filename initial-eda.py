# Importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# import the data of images
dataset_images = pd.read_csv("cancer-data/hmnist_28_28_L.csv")
print(dataset_images.head())
print(dataset_images.shape)

# Reading the data from HAM_metadata.csv
df_metadata = pd.read_csv("cancer-data/HAM10000_metadata.csv")
print(df_metadata.head())
print(df_metadata.shape)

# Plotting the disease class frequencies
sns.countplot(x = 'dx', data = df_metadata)
plt.xlabel('Disease', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Frequency Distribution of Classes', size=16)
plt.show()

# Plotting the frequencies of each age of those with the disease
bar, ax = plt.subplots(figsize=(10, 10))
sns.histplot(df_metadata['age'])
plt.title('Histogram of Age of Patients', size=16)
plt.show()

# Plotting the distribution of the localization of the disease
disease_location = df_metadata['localization'].value_counts()
plt.figure(figsize=(20, 5))
sns.countplot(x='localization', data=df_metadata)
plt.title('Disease localisation distribution')
plt.show()

# Plotting the gender frequencies of the patients
bar, ax = plt.subplots(figsize=(10, 10))
plt.pie(df_metadata['sex'].value_counts(),
        labels = df_metadata['sex'].value_counts().index, autopct="%.1f%%")
plt.title('Gender of Patient', size=16)
plt.show()
