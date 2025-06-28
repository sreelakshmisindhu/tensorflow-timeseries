import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from header import *

# The time dimension or the x-coordinate of the time series
TIME = np.arange(4 * 365 + 1, dtype="float32")

# Generating noisy data with seasonality and trend
y_intercept = 10
slope = 0.01
SERIES = trend(TIME, slope) + y_intercept
amplitude = 40
SERIES += seasonality(TIME, period=365, amplitude=amplitude)
noise_level = 2
SERIES += noise(TIME, noise_level, seed=42)

SPLIT_TIME = 1100
WINDOW_SIZE = 50

# Plot the series
#plt.figure(figsize=(10, 6))
#plot_series(TIME, SERIES)
#plt.show()

time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES, SPLIT_TIME)

#plt.figure(figsize=(10, 6))
#plot_series(time_train, series_train, title="Training")
#
#plt.figure(figsize=(10, 6))
#plot_series(time_valid, series_valid, title="Validation")
#
naive_forecast = np.concatenate(([series_train[-1]], series_valid[:-1]))  
print(f"validation series has shape: {series_valid.shape}\n")
print(f"naive forecast has shape: {naive_forecast.shape}\n")
print(f"comparable with validation series: {series_valid.shape == naive_forecast.shape}")

#plt.figure(figsize=(10, 6))
#plot_series(time_valid, series_valid, start=330, end=361, label="validation set")
#plot_series(time_valid, naive_forecast, start=330, end=361, label="naive forecast")
#plt.show()

print(f"Whole SERIES has {len(SERIES)} elements so the moving average forecast should have {len(SERIES)-50} elements")

# Try out your function
moving_avg = moving_average_forecast(SERIES, window_size=WINDOW_SIZE)
print(f"moving average forecast with whole SERIES has shape: {moving_avg.shape}\n")

# Slice it so it matches the validation period
moving_avg = moving_avg[1100 - WINDOW_SIZE:]
print(f"moving average forecast after slicing has shape: {moving_avg.shape}\n")
print(f"comparable with validation series: {series_valid.shape == moving_avg.shape}")
# Compute evaluation metrics
mse, mae = compute_metrics(series_valid, moving_avg)

diff_series = (SERIES[365:]-SERIES[:-365])
diff_time = TIME[365:]

print(f"Whole SERIES has {len(SERIES)} elements so the differencing should have {len(SERIES)-365} elements\n")
print(f"diff series has shape: {diff_series.shape}\n")
print(f"x-coordinate of diff series has shape: {diff_time.shape}\n")

diff_moving_avg = moving_average_forecast(diff_series, WINDOW_SIZE)
print(len(diff_series), len(diff_moving_avg), diff_series)
diff_moving_avg = diff_moving_avg[1100 -365 - WINDOW_SIZE:]
print(f"moving average forecast with diff series after slicing has shape: {diff_moving_avg.shape}\n")
print(f"comparable with validation series: {series_valid.shape == diff_moving_avg.shape}")

past_series = SERIES[1100-365:1096]
diff_moving_avg_plus_past = past_series + diff_moving_avg
print(f"past series has shape: {past_series.shape}\n")
print(f"moving average forecast with diff series plus past has shape: {diff_moving_avg_plus_past.shape}\n")
print(f"comparable with validation series: {series_valid.shape == diff_moving_avg_plus_past.shape}")

smooth_past_series = moving_average_forecast(SERIES[735-10:1096], 10)
diff_moving_avg_plus_smooth_past = smooth_past_series + diff_moving_avg

print(f"moving average forecast with diff series plus past has shape: {diff_moving_avg_plus_smooth_past.shape}\n")
print(f"comparable with validation series: {series_valid.shape == diff_moving_avg_plus_smooth_past.shape}")

plt.figure(figsize=(10, 6))

plot_series(time_valid, series_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)

#plot_series(time_valid, series_valid)
#plot_series(time_valid, diff_moving_avg_plus_past)

#plot_series(diff_time, diff_series)


#plot_series(time_valid, diff_series[1100 - 365:])
#plot_series(time_valid, diff_moving_avg)

#plot_series(time_valid, series_valid)
#plot_series(time_valid, moving_avg)
plt.show()
