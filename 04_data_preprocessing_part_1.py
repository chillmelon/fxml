#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import os


# In[2]:


resampled_dir = "../data/resampled/"
resampled_file_path = os.path.join(resampled_dir, "usdjpy-bar-2020-01-01-2024-12-31.pkl")
# resampled_file_path = os.path.join(resampled_dir, "usdjpy-bar-2020-01-01-2024-12-31.csv")


# In[3]:


df = pd.read_pickle(resampled_file_path)
df.head()


# In[ ]:





# In[4]:


import matplotlib.pyplot as plt

# Assuming your DataFrame is named df and 'timestamp' is datetime type
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['close'], label='Close Price')

plt.title('Close Price Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# # Sample smaller dataset

# df = df[df['timestamp'].dt.year == 2024]
# df.shape

# # Dealing with NaN value

# In[5]:


df.isna().sum()


# In[6]:


df = df.dropna()
df.isna().sum()


# In[7]:


df.shape


# # Time grouping

# By examining the data, we found that there are timeframes that only include Nan values. After clearing those NaN values, our time series data is no longer continuous, i.e. there're *time gaps*. Therefore, we need to group the data by checking their time continuity and assign *time_group* labels accordingly. This is a necessary process prior to the creation of sequences since *time gaps* can be huge and we don't want to create sequences or calculate indicators across them.

# In[8]:


# Ensure df is a copy (not a view)
df = df.copy()

# Then assign safely
df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.loc[:, 'time_delta'] = df['timestamp'].diff().dt.total_seconds()
df.loc[:, 'time_group'] = (df['time_delta'] != 60).cumsum().astype(int)
df = df.drop(columns='time_delta')


# In[9]:


df.isna().sum()


# In[10]:


df[['timestamp', 'time_group']]


# In[11]:


df['time_group'].value_counts().sort_values()


# In[12]:


import matplotlib.pyplot as plt

# Group by time_group and plot each with a different color
fig, ax = plt.subplots(figsize=(10,5))


ax.plot(df['timestamp'], df['close'])

ax.set_title('Close Price Over Time')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Close Price')
# ax.legend(title="time_group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[13]:


import matplotlib.pyplot as plt

# Group by time_group and plot each with a different color
fig, ax = plt.subplots(figsize=(10,5))

for group_id, group_data in df.groupby('time_group'):
    ax.plot(group_data['timestamp'], group_data['close'], label=f'Group {group_id}')

ax.set_title('Close Price Colored by time_group')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Close Price')
# ax.legend(title="time_group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[14]:


group_lengths = df['time_group'].value_counts()

plt.figure(figsize=(10, 5))
plt.hist(group_lengths, bins=50)
plt.title('Distribution of Time Group Lengths')
plt.xlabel('Number of Rows in Group')
plt.ylabel('Number of Groups')
plt.tight_layout()
plt.show()


# # Feature engineering

# ## Add delta and return

# The original price data is non-stationary, we can convert it into stationary data by calculating the difference between each timeframe as *delta* value.
# And we are also adding *return* values that indicates the percentage of grow/drop from the last timeframe.
# We will do the following process:
# 1. calculate delta and returns within each timegroup, note that this will result in adding 1 NaN value for each time group and will be dropped later on.
# 2. labeling the moving **direction** for each row based on the **return** and a given **threshold**
#     ```
#     'up' if x > threshold else ('down' if x < -threshold else 'flat')
#     ```
# 3. use a global encoder to encode the **direction**, this column will later on be used as target to train our model.

# In[15]:


GROUP_COl = 'time_group'


# In[16]:


import numpy as np

def add_delta(df, price_col: str = 'close', group_col: str = 'time_group') -> pd.DataFrame:
    df = df.copy()

    def calc(group):
        group[f"{price_col}_delta"] = group[price_col] - group[price_col].shift(1)
        group[f"{price_col}_return"] = group[price_col] / group[price_col].shift(1) - 1
        return group

    df = df.groupby(group_col, group_keys=False).apply(calc)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df.reset_index(drop=True)


# In[17]:


df = add_delta(df)


# In[18]:


# Plot the delta
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['close_delta'], label='Close Price Delta')
plt.axhline(0, color='gray', linestyle='--')  # zero line
plt.title('Delta of Close Prices')
plt.xlabel('Time')
plt.ylabel('Delta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[19]:


# Plot the return
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['close_return'], label='Close Price Future Return')
plt.axhline(0, color='gray', linestyle='--')  # zero line
plt.title('Return of Close Prices')
plt.xlabel('Time')
plt.ylabel('Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[20]:


df[1000:1200]


# In[21]:


df['time_group'].value_counts().sort_index()


# ## Plot histgram

# In[22]:


import seaborn as sns

df_copy = df.copy()

low, high = df_copy['close_delta'].quantile([0.025, 0.975])  # 95% range

plt.figure(figsize=(10, 5))
sns.histplot(df_copy['close_delta'])
plt.title("Distribution of Close Price Delta(Zoomed)")
plt.xlabel("Close Price Delta")
plt.ylabel("Frequency")
plt.xlim(low, high)
plt.tight_layout()
plt.show()


# In[23]:


df_copy = df.copy()

low, high = (df_copy['close_return']*1000).quantile([0.025, 0.975])  # 95% range

plt.figure(figsize=(10, 5))
sns.histplot(df_copy['close_return']*1000)
plt.title("Distribution of Close Percentage Change (Zoomed)")
plt.xlabel("Close Price Percentage Change")
plt.ylabel("Frequency")
plt.xlim(low, high)
plt.tight_layout()
plt.show()



# In[24]:


df.isna().sum()


# In[25]:


df = df.dropna()
df.isna().sum()


# ## Classification Labeling

# ## Add direction

# In[26]:


def add_direction(df, delta_columns=['close'], threshold=0.005):
    """
    Add directional class labels based on deltas and a threshold.
    """
    df = df.copy()

    for col in delta_columns:
        df[f"{col}_direction"] =  df[f"{col}_return"].apply(lambda x: 'up' if x > threshold else ('down' if x < -threshold else 'flat'))
    return df


# In[27]:


df = add_direction(df, threshold=3e-5)


# In[28]:


df['close_direction'].value_counts()


# In[29]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Slice and shift for next movement direction
df_filtered = df[-101:].copy()
df_filtered['next_direction'] = df_filtered['close_direction'].shift(-1)
df_filtered = df_filtered[:-1]  # Drop the last row where next_direction is NaN

# Plot
plt.figure(figsize=(10, 5))

# Color map
colors = df_filtered['next_direction'].map({'up': 'green', 'down': 'red', 'flat': 'gray'})

# Plot scatter and line
plt.scatter(df_filtered.index, df_filtered['close'], c=colors, s=15)
plt.plot(df_filtered.index, df_filtered['close'], color='black', linewidth=0.5, alpha=0.5)

# Create manual legend
legend_patches = [
    mpatches.Patch(color='green', label='Next Up'),
    mpatches.Patch(color='red', label='Next Down'),
    mpatches.Patch(color='gray', label='No Change'),
]
plt.legend(handles=legend_patches, title='Next Movement')

# Final formatting
plt.title('Close Price with Movement Direction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[30]:


import matplotlib.pyplot as plt

# Slice and compute next direction
df_filtered = df[-101:].copy()
df_filtered['next_direction'] = df_filtered['close_direction'].shift(-1)
df_filtered = df_filtered[:-1]

# Map direction to y-axis position and color
direction_to_y = {'down': -1, 'flat': 0, 'up': 1}
direction_to_color = {'up': 'green', 'down': 'red', 'flat': 'gray'}

df_filtered['direction_y'] = df_filtered['next_direction'].map(direction_to_y)
colors = df_filtered['next_direction'].map(direction_to_color)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(df_filtered['timestamp'], df_filtered['direction_y'], c=colors, s=20)

# Customize y-axis with labels
plt.yticks([-1, 0, 1], ['Down', 'Flat', 'Up'])

# Format time axis
plt.xlabel('Time')
plt.ylabel('Direction')
plt.title('Next Price Movement Direction (Scatter)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[31]:


df.head()


# ## Label encoding

# ## One Hot Encoding

# In[32]:


# One-hot encode with get_dummies
one_hot = pd.get_dummies(df['close_direction'], prefix='prob').astype('float32')

df = df.join(one_hot)

df.head


# In[33]:


from sklearn.preprocessing import LabelEncoder

def add_label(df, class_col='direction'):
    """
    Add directional class labels based on deltas and a threshold.
    """
    df = df.copy()

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[class_col])
    direction_counts = df["label"].value_counts()
    print(direction_counts)
    print(label_encoder.classes_)
    return df, label_encoder


# In[34]:


df, encoder = add_label(df, class_col='close_direction')


# In[35]:


import joblib

ENCODER_PATH = '../data/processed/label_encoder.pkl'

joblib.dump(encoder, ENCODER_PATH)


# In[36]:


df.head()


# # Timegroup Filtering

# Before creating sequences, we have to make sure every **time group** have enough data to create at least **1** sequence.
# Given the sequence length is X, horizon is H, each time group must have at least X + H timeframes. 

# In[37]:


SEQ_LEN = 30
HORIZON = 1

min_len = SEQ_LEN + HORIZON


# In[38]:


lengths = df['time_group'].value_counts()
before = lengths
after = lengths[lengths >= 100]

plt.figure(figsize=(10, 5))
plt.hist([before, after], bins=50, label=['Before Filter', 'After Filter'], stacked=False, log=True)
plt.title('Time Group Lengths: Before vs After Filter')
plt.xlabel('Group Length')
plt.ylabel('Number of Groups')
plt.legend()
plt.tight_layout()
plt.show()


# In[39]:


df = df.groupby("time_group").filter(lambda g: len(g) >= min_len)


# In[40]:


df['time_group'].nunique()


# In[41]:


df['time_group'].value_counts()


# # Saving the file

# In[42]:


PROCESSED_DIR = "../data/processed/"
PROCESSED_FILENAME = "usdjpy-bar-2020-01-01-2024-12-31_processed.pkl"
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DIR, PROCESSED_FILENAME)

df.to_pickle(PROCESSED_FILE_PATH)


# In[ ]:




