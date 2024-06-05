import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"E:\Tugas\Numerik\Bahan\Student_Performance.csv"
data = pd.read_csv(file_path)

# Check the structure of the dataset
print(data.head())

# Extract relevant columns
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

# Define the linear model
def linear_model(x, a, b):
    return a + b * x

# Define the power model
def power_model(x, a, b):
    return a * np.power(x, b)

# Fit the linear model
params_linear, _ = curve_fit(linear_model, NL, NT)
a_linear, b_linear = params_linear

# Fit the power model
params_power, _ = curve_fit(power_model, NL, NT, maxfev=10000)
a_power, b_power = params_power

# Predict using both models
NT_pred_linear = linear_model(NL, a_linear, b_linear)
NT_pred_power = power_model(NL, a_power, b_power)

# Calculate RMS errors
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_power = np.sqrt(mean_squared_error(NT, NT_pred_power))

# Plotting the results
plt.figure(figsize=(14, 6))

# Plot for Linear Model
plt.subplot(1, 2, 1)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_linear, color='red', label='Linear Fit')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Linear Model')
plt.legend()

# Plot for Power Model
plt.subplot(1, 2, 2)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_power, color='green', label='Power Fit')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Power Model')
plt.legend()

plt.show()

# Output the results
print(f'Linear Model Parameters: a = {a_linear}, b = {b_linear}')
print(f'Power Model Parameters: a = {a_power}, b = {b_power}')
print(f'RMS Error for Linear Model: {rms_linear}')
print(f'RMS Error for Power Model: {rms_power}')
