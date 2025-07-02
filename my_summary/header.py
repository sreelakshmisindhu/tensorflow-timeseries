import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#def plot_series(time, series, format="-", title="", start=0, end=None, label=None):
#    """
#    Visualizes time series data
#      time (array of int), series (array of int), format (string), start (int), end (int), label (list of strings)
#    """
#    plt.figure(figsize=(10, 6))
#    plt.plot(time[start:end], series[start:end], format)
#    plt.xlabel("Time")
#    plt.ylabel("Value")
#    plt.title(title)
#    if label:
#      plt.legend(fontsize=14, labels=label)
#    plt.grid(True)
#    plt.show()


def plot_series(time, series, format="-", title="", label=None, start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)

def trend(time, slope=0):
    """
    Generates synthetic data that follows a straight line given a slope value.
      time (array of int), slope (float), series (array of float) 
    """
    series = slope * time
    return series

def seasonal_pattern(season_time):
    """
    Just an arbitrary pattern
      season_time (array of float),  data_pattern (array of float) 
    """
    data_pattern = np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
    
    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    """
    Repeats the same pattern at each period
      time (array of int), period (int), amplitude (int), phase (int), data_pattern (array of float)
    """
    
    # Define the measured values per period
    season_time = ((time + phase) % period) / period
    # Generates the seasonal data scaled by the defined amplitude
    data_pattern = amplitude * seasonal_pattern(season_time)

    return data_pattern

def noise(time, noise_level=1, seed=None):
    """Generates a normally distributed noisy signal
      time (array of int), noise_level (float), seed (int), noise (array of float)
    """
   # Generate a random number for each time step and scale by the noise level
    rnd = np.random.RandomState(seed) 
    noise = rnd.randn(len(time)) * noise_level
    
    return noise

def autocorrelation(time, amplitude, seed=None):
    """
    Generates autocorrelated data
      time (array of int), amplitude (float), seed (int), ar (array of float)
    """
    # Initialize array of random numbers equal to the length 
    # of the given time steps plus an additional step
    rnd = np.random.RandomState(seed)
    ar = rnd.randn(len(time) + 1)

    # Define scaling factor
    phi = 0.8

    # Autocorrelate element 11 onwards with the measurement at 
    # (t-1), where t is the current time step
    for step in range(1, len(time) + 1):
        ar[step] += phi * ar[step - 1]
    
    # Get the autocorrelated data and scale with the given amplitude.
    ar = ar[1:] * amplitude
    
    return ar

def impulses(time, num_impulses, amplitude=1, seed=None):
    """
    Generates random impulses
      time (array of int), num_impulses (int), amplitude (float), seed (int), series (array of float)
    """
    # Generate random numbers
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=num_impulses)
    series = np.zeros(len(time))
    # Insert random impulses
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude

    return series

def autocorrelation_impulses(source, phis):
    """
    Generates autocorrelated data from impulses
      source (array of float), phis (dict), ar (array of float)
    """

    # Copy the source
    ar = source.copy()

    # Compute new series values based on the lag times and decay rates
    for step, value in enumerate(source):
        for lag, phi in phis.items():
            if step - lag > 0:
              ar[step] += phi * ar[step - lag]

    return ar

def train_val_split(time, series, split_time):
    """
	Split time series into train and validation sets
        time (np.ndarray), series (np.ndarray) returns: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    time_train = time[:split_time]
    series_train = series[:split_time]
    time_valid = time[split_time:]
    series_valid = series[split_time:]

    return time_train, series_train, time_valid, series_valid

def compute_metrics(true_series, forecast):
    """compute mean squared error and mean absolute error for prediction
        true_series (np.ndarray), forecast (np.ndarray), returns:(np.float64, np.float64): MSE and MAE
    """
    mse = tf.keras.metrics.mse(true_series, forecast).numpy()
    mae = tf.keras.metrics.mae(true_series, forecast).numpy()

    return mse, mae

def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
        If window_size=1, then this is equivalent to naive forecast
	 series (np.ndarray), window_size (int), Returns: np.ndarray
    """
    forecast = []

    np_forecast = np.array(
        [ series[t - window_size : t].mean()     # mean of the *previous* window_size values
          for t in range(window_size, len(series)) ]
    )
    
    return np_forecast

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float), window_size (int), batch_size (int), shuffle_buffer(int), Return: dataset (TF Dataset)
    """
  
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)
    # Create batches of windows
    dataset = dataset.batch(batch_size)
    
    # Optimize the dataset for training
    #dataset = dataset.cache().prefetch(1)
    
    return dataset

def generate_time_series():
    """ Creates timestamps and values of the time series """
    
    # The time dimension or the x-coordinate of the time series
    time = np.arange(4 * 365 + 1, dtype="float32")

    # Initial series is just a straight line with a y-intercept
    y_intercept = 10
    slope = 0.005
    series = trend(time, slope) + y_intercept

    # Adding seasonality
    amplitude = 50
    series += seasonality(time, period=365, amplitude=amplitude)

    # Adding some noise
    noise_level = 3
    series += noise(time, noise_level, seed=51)
    
    return time, series
