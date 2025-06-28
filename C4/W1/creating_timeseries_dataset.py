import numpy as np
from header import * #autocorrelation, trend 

time = np.arange(365)
# Generate autocorrelated data with an upward trend
series = autocorrelation(time, 10, seed=42) + trend(time, 2)

# Plot the results
plot_series(time[:200], series[:200])
