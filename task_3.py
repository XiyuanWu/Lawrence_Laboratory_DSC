# %% [markdown]
# # Task 3 Activation Map Reconstruction

# %%
# Analysis
import numpy as np
import pandas as pd

import glob, re, os
from typing import List

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# # Evaluate
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# # Data processing
# from sklearn.preprocessing import StandardScaler
# from sklearn.calibration import LabelEncoder

from sklearn.utils import shuffle

# Save model
import pickle

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 100)
plt.style.use('ggplot')

# import python scripts
# from ./cardiac_challenge/notebooks/cardiac_ml_tools.py
import sys
sys.path.append('./cardiac_challenge/notebooks')
from cardiac_ml_tools import read_data_dirs
from cardiac_ml_tools import get_standard_leads
from cardiac_ml_tools import get_activation_time

# %%
# Run the cardiac_ml_tools.py script
# %run ./cardiac_challenge/notebooks/cardiac_ml_tools.py

# %%
# Load the dataset
data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR = './cardiac_challenge/intracardiac_dataset/' # path to the intracardiac_dataset

for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(DIR + x)
file_pairs = read_data_dirs(data_dirs)
print('Number of file pairs: {}'.format(len(file_pairs)))
# example of file pair
print("Example of file pair:")
print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))



# %%
# If file active.npy and ecg.npy are already created, load them
if os.path.exists('./combine_dataset/ecg_data.npy') and os.path.exists('./combine_dataset/active_time.npy'):
    # ECGData = np.load('./combine_dataset/ecg_data.npy')
    # ActTime = np.load('./combine_dataset/active_time.npy')
    ECGData = np.load('./combine_dataset/ecg_data_5000.npy')
    ActTime = np.load('./combine_dataset/active_time_5000.npy')

else:
    # file_pairs is a list where each element is a tuple containing the file paths for ECG data and activation time data
    num_samples = 16117  # Number of samples to process
    num_timesteps = 500  # Each ECG data has 500 timesteps
    num_leads = 12  # Standard ECG leads count after processing

    # Initialize arrays to store combined data
    ECGData = np.zeros((num_samples, num_timesteps * num_leads))  # Flattened array for 12 leads data
    ActTime = np.zeros((num_samples, 75))  # Store 75 activation times per sample

    # Process each sample
    for i in range(num_samples):
        # Load ECG data
        pECGData = np.load(file_pairs[i][0])
        pECGData = get_standard_leads(pECGData)  # Convert to 12 standard leads
        ECGData[i, :] = pECGData.flatten()  # Flatten and store in the combined array

        # Load activation time data
        VmData = np.load(file_pairs[i][1])
        ActTime[i, :] = get_activation_time(VmData).flatten()
        # ActTime[i, :] = ActTime.flatten()  # Flatten the (75, 1) array to fit into (75,) array

    # Create directory if it does not exist
    output_dir = './combine_dataset'
    os.makedirs(output_dir, exist_ok=True)

    # Save combined datasets to .npy format
    np.save(os.path.join(output_dir, 'ecg_data.npy'), ECGData)
    np.save(os.path.join(output_dir, 'active_time.npy'), ActTime)


# %% [markdown]
# After combine, check them

# %%
print("ECGData shape: {}".format(ECGData.shape))
print("ActTime shape: {}".format(ActTime.shape))

# %% [markdown]
# 
# #### 3.1.2 Split Dataset

# %%
# Shuffle indices
indices = np.arange(ECGData.shape[0])
shuffled_indices = shuffle(indices, random_state=42)

# Define the split point
split_ratio = 0.7  # 70% training, 30% test
split_point = int(len(shuffled_indices) * split_ratio)

# Split indices into training and test sets
train_indices = shuffled_indices[:split_point]
test_indices = shuffled_indices[split_point:]

# Use indices to create training and test data
X_train = ECGData[train_indices]
y_train = ActTime[train_indices]
X_test = ECGData[test_indices]
y_test = ActTime[test_indices]

# %%
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %% [markdown]
# ### 3.2 Modeling
# 
# #### 3.2.1 Define the 1D CNN Model

# %%
'''
class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()

        # convolutional layers -> relu -> convolutional layers -> relu -> 
        # pooling -> flatten -> fully connected layers
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        # self.relu4 = nn.ReLU()

        self.pool = nn.MaxPool1d(3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(21504, 512)
        # self.fc1 = nn.Linear(43008, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 75)  # Output the activation times

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)

        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
'''

class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=25, padding = 12)
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization after conv1
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, padding = 7)
        self.bn2 = nn.BatchNorm1d(128)  # Batch Normalization after conv2
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, padding = 4)
        self.bn3 = nn.BatchNorm1d(256)  # Batch Normalization after conv3
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=74, kernel_size=3, padding = 1)
        self.bn4 = nn.BatchNorm1d(512)  # Batch Normalization after conv4

        # self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding = 1)
        # self.bn5 = nn.BatchNorm1d(1024)  # Batch Normalization after conv4
        # self.conv6 = nn.Conv1d(in_channels=1024, out_channels=75, kernel_size=3, padding = 1)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.sigmoid(x)

        return x

# %% [markdown]
# #### 3.2.2 Initialize the Model and Optimizer


if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


# %%
model = Simple1DCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Configure Loss Function
criterion = nn.MSELoss()

# %% [markdown]
# #### 3.2.3 Prepare & Train Model

# %%
# Prepare
X_train_tensor = torch.tensor(X_train.reshape(3500, 12, 500), dtype=torch.float32)  # Add channel dimension
# X_train_tensor = torch.tensor(X_train.reshape(3500, 12, 500), dtype=torch.float32)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Train
num_epochs = 50  # or however many you deem necessary

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# %% [markdown]
# ### 3.3 Evaluate the model

# %%
# Evaluate model
# model.eval()

# Prepare test data
# X_test_tensor = torch.tensor(X_test.reshape(4836, 12, 500), dtype=torch.float32)  # Add channel dimension
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# # Predict
# with torch.no_grad():
#     y_pred = model(X_test_tensor)

# # Calculate loss
# test_loss = criterion(y_pred, y_test_tensor)
# print(f'Test Loss: {test_loss.item()}')

# %%
# Save the model
# torch.save(model.state_dict(), '1dcnn.pth')


