import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

# Load the data from CSV file
df = pd.read_csv('raw_data.csv')  # Replace 'your_data.csv' with your actual file name
x_original = np.arange(len(df))  # Assuming original data points are equally spaced
y_original = df.iloc[:, 1].values  # First column of the CSV

# Create new x values for interpolation
x_new = np.linspace(0, len(df) - 1, 200)  # Create 200 equally spaced new x values

# Perform cubic spline interpolation
cs = CubicSpline(x_original, y_original)
y_new = cs(x_new)

# Save the interpolated data to a new CSV file
interpolated_data = pd.DataFrame({'cubic': y_new})
interpolated_data.to_csv('cubic.csv', index=False)