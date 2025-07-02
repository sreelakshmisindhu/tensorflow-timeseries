import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import header

# Parameters

def DNN_model(window_size):

    model = tf.keras.models.Sequential([ 
    tf.keras.Input(shape=(window_size,)),
    tf.keras.layers.Dense(8, activation="relu"),
    #tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1)
        
    ]) 

    #model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(1e-3)) #SGD(learning_rate=1e-6, momentum=0.95)) "mse"
    return model

def RNN_model(window_size):
    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(window_size,1)),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])
    return model


split_time = 1100
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
time, series = header.generate_time_series()
time_train, series_train, time_valid, series_valid = header.train_val_split(time, series, split_time)
dataset = header.windowed_dataset(series_train, window_size, batch_size, shuffle_buffer_size)

for windows in dataset.take(1):
  print(f'data type: {type(windows)}')
  print(f'number of elements in the tuple: {len(windows)}')
  print(f'shape of first element: {windows[0].shape}')
  print(f'shape of second element: {windows[1].shape}')


# Build the NN
#model_baseline = tf.keras.models.Sequential([
#    tf.keras.Input(shape=(window_size,)),
#    #tf.keras.layers.Dense(6, activation="relu"), 
#    tf.keras.layers.Dense(8, activation="relu"), 
#    tf.keras.layers.Dense(1)
#])

model_baseline = RNN_model(window_size)
# TRAINING WITH FIXED LEARNING RATE
model_baseline.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
history = model_baseline.fit(dataset,epochs=100)
model_baseline.summary()
loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()


## TUNING LEARNING RATE uncomment the following only for tuning
#lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#    lambda epoch: 1e-8 * 10**(epoch / 20))
#optimizer = tf.keras.optimizers.SGD(momentum=0.9)
#model_baseline.compile(loss="mse", optimizer=optimizer)
#history = model_baseline.fit(dataset, epochs=100, callbacks=[lr_schedule])
## Define the learning rate array
#lrs = 1e-8 * (10 ** (np.arange(100) / 20))
#plt.figure(figsize=(10, 6))
#plt.grid(True)
#plt.semilogx(lrs, history.history["loss"])
#plt.tick_params('both', length=10, width=1, which='both')
#plt.axis([1e-8, 1e-3, 0, 300])
#plt.savefig("learning_rate_tuning.png")
##pick a point in a downward slope, here the network is still learning and is stable
##Choose close to the minimum point of the graph, the training converges  quicker.
# jagged edges and pointing upwards the training to become unstable



## Print the initial layer weights
#print("Layer weights: \n {} \n".format(model_baseline.get_weights()))

# Print the layer weights
print("Layer weights {}".format(model_baseline.get_weights()))

# Shape of the first 20 data points slice
print(f'shape of series[0:20]: {series[0:20].shape}')

# Shape after adding a batch dimension
print(f'shape of series[0:20][np.newaxis]: {series[0:20][np.newaxis].shape}')

# Shape after adding a batch dimension (alternate way)
print(f'shape of series[0:20][np.newaxis]: {np.expand_dims(series[0:20], axis=0).shape}')

# Sample model prediction
print(f'model prediction: {model_baseline.predict(series[0:20][np.newaxis])}')

# Initialize a list
forecast = []

# Reduce the original series
forecast_series = series[split_time - window_size:]
# Use the model to predict data points per window size
for time in range(len(forecast_series) - window_size):
  forecast.append(model_baseline.predict(forecast_series[time:time + window_size][np.newaxis], verbose=0))


# Compare number of elements in the predictions and the validation set
print(f'length of the forecast list: {len(forecast)}')
print(f'shape of the validation set: {series_valid.shape}')


# Preview shapes after using the conversion and squeeze methods
print(f'shape after converting to numpy array: {np.array(forecast).shape}')
print(f'shape after squeezing: {np.array(forecast).squeeze().shape}')

# Convert to a numpy array and drop single dimensional axes
results = np.array(forecast).squeeze()

# Compute the metrics
print(tf.keras.metrics.mse(series_valid, results).numpy())
print(tf.keras.metrics.mae(series_valid, results).numpy())


# Plot the results
plt.figure(figsize=(10, 6))
#header.plot_series(time, series)

# Overlay the results with the validation set
header.plot_series(time_valid, series_valid)
header.plot_series(time_valid, results)
plt.show()

