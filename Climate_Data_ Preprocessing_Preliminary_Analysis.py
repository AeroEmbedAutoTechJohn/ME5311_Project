#!/usr/bin/env python
# coding: utf-8

# # ME5311-PROJECT:
# # Climate Data Analysis: Preprocessing and Preliminary Analysis of Sea Level Pressure and Two-Meter Temperature Data

# In[82]:


# Import necessary libraries
import os
import xarray as xr
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# dimensions of data

# In[83]:


n_samples = 16071
n_latitudes = 101
n_longitudes = 161
shape = (n_samples, n_latitudes, n_longitudes)


# In[84]:


path = 'data'
slp_path = os.path.join(path,'slp.nc')
t2m_path = os.path.join(path,'t2m.nc')

# load data
ds_slp = xr.open_dataset(slp_path)  # Load sea level pressure data
print(f"ds_slp:{ds_slp}")
ds_t2m = xr.open_dataset(t2m_path)  # Load two-meter temperature data
print(f"ds_t2m:{ds_t2m}")


# get data values for sea level pressure

# In[85]:


da_slp_msl = ds_slp['msl']  # 'msl' is the variable name for sea level pressure
x_slp = da_slp_msl.values
print(x_slp)
# print(x_slp.shape)


# get data values for two-meter temperature

# In[86]:


da_t2m_t2m = ds_t2m['t2m']  # 't2m' is the variable name for two-meter temperature
x_t2m = da_t2m_t2m.values
print(x_t2m)
# print(x_t2m.shape)


# get time snapshots from sea level pressure dataset

# In[87]:


da_slp_time = ds_slp['time']
t_slp = da_slp_time.values
print(t_slp)


# get time snapshots from two-meter temperature dataset

# In[88]:


da_t2m_time = ds_t2m['time']
t_t2m = da_t2m_time.values
print(t_t2m)


# get longitude values from sea level pressure dataset

# In[89]:


da_slp_longitude = ds_slp['longitude']
lon_slp = da_slp_longitude.values
print(lon_slp)


# get latitude values from sea level pressure dataset

# In[90]:


da_slp_latitude = ds_slp['latitude']
lat_slp = da_slp_latitude.values
print(lat_slp)


# ONLY if not enough memory

# In[91]:


# OPTIONAL: Reduce resolution if not enough memory
# Reduce resolution for sea level pressure dataset
# low_res_ds_slp = ds_slp[{'longitude': slice(None, None, 2), 'latitude': slice(None, None, 2)}]
# low_res_ds_slp.to_netcdf(path='slp_low_res.nc')
#
# # Reduce resolution for two-meter temperature dataset
# low_res_ds_t2m = ds_t2m[{'longitude': slice(None, None, 2), 'latitude': slice(None, None, 2)}]
# low_res_ds_t2m.to_netcdf(path='t2m_low_res.nc')


# Calculate and print basic statistical description for sea level pressure

# In[92]:


print(f"SLP Mean: {da_slp_msl.mean().values}")
print(f"SLP Std: {da_slp_msl.std().values}")
print(f"SLP Min: {da_slp_msl.min().values}")
print(f"SLP Max: {da_slp_msl.max().values}")


# Calculate and print basic statistical description for two-meter temperature

# In[93]:


print(f"T2M Mean: {da_t2m_t2m.mean().values}")
print(f"T2M Std: {da_t2m_t2m.std().values}")
print(f"T2M Min: {da_t2m_t2m.min().values}")
print(f"T2M Max: {da_t2m_t2m.max().values}")


# Print time range for sea level pressure dataset

# In[94]:


print("SLP Time Range:", ds_slp['time'].min().values, "to", ds_slp['time'].max().values)
time_list = ds_slp['time'].values
# 通过遍历打印每个时间点
for t in time_list:
    print(t)


# Print time range for two-meter temperature dataset

# In[95]:


print("T2M Time Range:", ds_t2m['time'].min().values, "to", ds_t2m['time'].max().values)
time_list = ds_t2m['time'].values
# 通过遍历打印每个时间点
for t in time_list:
    print(t)


# # Visualization Check
# 
# Visualization Check for Sea Level Pressure

# In[96]:


# Select data for the first time point for plotting
time_index = 0
# Plot the spatial distribution of Sea Level Pressure (SLP)
plt.figure(figsize=(16, 10))
plt.contourf(lon_slp, lat_slp, x_slp[time_index, :, :], cmap='viridis', levels=np.linspace(x_slp.min(), x_slp.max(), 20))
plt.colorbar(label='Sea Level Pressure (Pa)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Sea Level Pressure at the First Time Point')
plt.show()

# Plot the spatial distribution of Sea Level Pressure (SLP) for the last time point
time_index = -1
plt.figure(figsize=(16, 10))
plt.contourf(lon_slp, lat_slp, x_slp[time_index, :, :], cmap='viridis', levels=np.linspace(x_slp.min(), x_slp.max(), 20))
plt.colorbar(label='Sea Level Pressure (Pa)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Sea Level Pressure at the Last Time Point')
plt.show()


# Visualization Check for Two-Meter Temperature

# In[97]:


# Select data for the first time point for plotting
time_index = 0
# Plot the spatial distribution of Two-Meter Temperature (T2M)
plt.figure(figsize=(16, 10))
plt.contourf(lon_slp, lat_slp, x_t2m[time_index, :, :], cmap='RdYlBu_r', levels=np.linspace(x_t2m.min(), x_t2m.max(), 20))
plt.colorbar(label='Two-meter Temperature (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Two-Meter Temperature at the First Time Point')
plt.show()

# Plot the spatial distribution of Two-Meter Temperature (T2M) for the last time point
time_index = -1
plt.figure(figsize=(16, 10))
plt.contourf(lon_slp, lat_slp, x_t2m[time_index, :, :], cmap='RdYlBu_r', levels=np.linspace(x_t2m.min(), x_t2m.max(), 20))
plt.colorbar(label='Two-meter Temperature (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Two-Meter Temperature at the Last Time Point')
plt.show()


# In[98]:


latitude_index = 0
longitude_index = 0

# Set the size of the chart
plt.figure(figsize=(45, 20))
# Plot the time series of Sea Level Pressure (SLP)
plt.plot(da_slp_time, x_slp[:, latitude_index, longitude_index], label='SLP')
# Set the range of the X-axis from 1979 to 2022
start_date = pd.to_datetime('1978-12-31')
end_date = pd.to_datetime('2023-01-01')
plt.xlim(start_date, end_date)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Automatically rotate date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.xlabel('Time')
plt.ylabel('Sea Level Pressure')
plt.title('Sea Level Pressure (SLP) Time Series at the First Geographical Position')
plt.legend()
plt.show()

# Set the size of the chart
plt.figure(figsize=(45, 20))
# Plot the time series of Two-Meter Temperature (T2M)
plt.plot(da_t2m_time, x_t2m[:, latitude_index, longitude_index], label='T2M', color='r')
# Set the range of the X-axis from 1979 to 2022
start_date = pd.to_datetime('1978-12-31')
end_date = pd.to_datetime('2023-01-01')
plt.xlim(start_date, end_date)
# Set the date format on the X-axis for every year and ensure each year is marked
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Automatically rotate date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.xlabel('Time')
plt.ylabel('Two-meter Temperature (K)')
plt.title('Two-Meter Temperature (T2M) Time Series at the First Geographical Position')
plt.legend()
plt.show()


# In[99]:


# Select the center latitude and longitude indices
latitude_index = n_latitudes // 2
longitude_index = n_longitudes // 2

# Set the size of the chart
plt.figure(figsize=(45, 20))
# Plot the time series of Sea Level Pressure (SLP)
plt.plot(da_slp_time, x_slp[:, latitude_index, longitude_index], label='SLP')
# Set the range of the X-axis from 1979 to 2022
start_date = pd.to_datetime('1978-12-31')
end_date = pd.to_datetime('2023-01-01')
plt.xlim(start_date, end_date)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Automatically rotate date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.xlabel('Time')
plt.ylabel('Sea Level Pressure')
plt.title('Long-term Sea Level Pressure (SLP) Time Series Analysis at the Dataset\'s Central Geographical Location')
plt.legend()
plt.show()

# Set the size of the chart
plt.figure(figsize=(45, 20))
# Plot the time series of Two-Meter Temperature (T2M)
plt.plot(da_t2m_time, x_t2m[:, latitude_index, longitude_index], label='T2M', color='r')
# Set the range of the X-axis from 1979 to 2022
start_date = pd.to_datetime('1978-12-31')
end_date = pd.to_datetime('2023-01-01')
plt.xlim(start_date, end_date)
# Set the date format on the X-axis for every year and ensure each year is marked
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Automatically rotate date labels to avoid overlap
plt.gcf().autofmt_xdate()
plt.xlabel('Time')
plt.ylabel('Two-meter Temperature (K)')
plt.title('Long-term Two-Meter Temperature (T2M) Time Series Analysis at the Dataset\'s Central Geographical Location')
plt.legend()
plt.show()


# In[100]:


# Analyzing data for the first geographical location
slp_series = x_slp[:, 0, 0]
t2m_series = x_t2m[:, 0, 0]

# Calculate Pearson correlation coefficient
corr_coefficient, p_value = pearsonr(slp_series, t2m_series)
print(f"Pearson Correlation Coefficient: {corr_coefficient}, P-value: {p_value}")
# Perform linear regression
slope, intercept, r_value, p_value_reg, std_err = linregress(slp_series, t2m_series)
# Generate x and y values for the regression line
x_values = np.linspace(slp_series.min(), slp_series.max(), 500)  # Generate enough x values for a smooth line
y_values = intercept + slope * x_values
# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(slp_series, t2m_series, alpha=0.3, edgecolors='w', label=f'Pearson r: {corr_coefficient:.2f}, p-value: {p_value:.2e}')
# Plot regression line
plt.plot(x_values, y_values, color='red', label=f'Linear Regression\ny = {intercept:.2f} + {slope:.2f}x\nR-squared: {r_value**2:.2f}')
# Add chart title and axis labels
plt.title('Correlation and Linear Regression Analysis between SLP and T2M at the First Geographical Location')
plt.xlabel('Sea Level Pressure')
plt.ylabel('Two-meter Temperature')
plt.legend()
# Show chart
plt.show()


# In[101]:


# Convert time to pandas datetime
times_pd = pd.to_datetime(t_slp)

# Select a specific location, the center of the dataset
lat_index = n_latitudes // 2
lon_index = n_longitudes // 2

# Extract temperature and pressure data for the center point
temp_center = ds_t2m['t2m'][:, lat_index, lon_index].groupby('time.month').mean()
pressure_center = ds_slp['msl'][:, lat_index, lon_index].groupby('time.month').mean()

# Plot the chart
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature (K)', color=color)
ax1.plot(temp_center.month, temp_center, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Pressure (Pa)', color=color)
ax2.plot(pressure_center.month, pressure_center, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Otherwise, the right y-label might be slightly clipped
plt.title('Seasonal Variation of Temperature and Pressure at the Dataset\'s Center Location')
plt.show()


# In[102]:


lat_index = 0
lon_index = 0

temperature = ds_t2m['t2m'][:, lat_index, lon_index].values
pressure = ds_slp['msl'][:, lat_index, lon_index].values

# Time processing
t_slp = pd.to_datetime(ds_slp['time'].values)

# Creating DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure
}, index=t_slp)

# Calculate the annual average temperature and pressure
annual_avg = df.resample('Y').mean()

# Calculate regression line parameters
temp_regress = linregress(annual_avg.index.year, annual_avg['Temperature'])
press_regress = linregress(annual_avg.index.year, annual_avg['Pressure'])

# Drawing the chart, including regression lines
fig, ax1 = plt.subplots(figsize=(45, 20))

color = 'tab:red'
ax1.set_xlabel('Year', fontsize=20)  # Increase font size of x-axis labels
ax1.set_ylabel('Temperature (°C)', color=color, fontsize=20)  # Increase font size of y-axis labels
ax1.plot(annual_avg.index.year, annual_avg['Temperature'], color=color, label='Temperature')
ax1.plot(annual_avg.index.year, temp_regress.intercept + temp_regress.slope * annual_avg.index.year,
         color='tab:orange', linestyle='--', label='Temperature Regression Line')
ax1.tick_params(axis='y', labelcolor=color, labelsize=20)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Pressure (hPa)', color=color, fontsize=20)
ax2.plot(annual_avg.index.year, annual_avg['Pressure'], color=color, label='Pressure')
ax2.plot(annual_avg.index.year, press_regress.intercept + press_regress.slope * annual_avg.index.year,
         color='tab:purple', linestyle='--', label='Pressure Regression Line')
ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

years = np.arange(1979, 2023)  # From 1979 to 2022
ax1.set_xticks(years)
ax1.set_xticklabels(years, fontsize=20)

fig.tight_layout()
plt.title('Annual Average Temperature and Pressure with Regression Lines (1979-2022)', fontsize=22)
ax1.legend(loc="upper left", fontsize=20)
ax2.legend(loc="upper right", fontsize=20)
plt.show()


# In[102]:




