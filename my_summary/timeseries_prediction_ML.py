import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from header import *

# Save all global variables
SPLIT_TIME = 1100
WINDOW_SIZE = 20
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
TIME, SERIES = generate_time_series()

plt.figure(figsize=(10, 6))
#plot_series(TIME, SERIES)
time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES, SPLIT_TIME)
test_dataset = windowed_dataset(series_train, 10, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)

# Get the first batch of the test dataset
batch_of_features, batch_of_labels = next((iter(test_dataset)))

print(f"batch_of_features has shape: {batch_of_features.shape}\n")
print(f"batch_of_labels has shape: {batch_of_labels.shape}\n")

plt.plot(np.arange(10), batch_of_features[0].numpy(), label='features')
plt.plot(np.arange(9,11), [batch_of_features[0].numpy()[-1], batch_of_labels[0].numpy()], label='label');
plt.legend()

plt.show()


