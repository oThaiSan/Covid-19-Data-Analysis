import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds


###################################################### DEATHS DATASET ######################################################

# Load the DEATHS dataset
df = pd.read_csv('time_series_covid19_deaths_US.csv')
df = df.fillna(0)

# Define the columns that uniquely identify each row but aren't part of the time series data
id_vars = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key']

# Columns that are the time series data (i.e., the dates)
date_columns = df.columns[df.columns.get_loc('1/22/20'):]

# Melt the DataFrame to make 'Date' and 'Deaths' columns
df_long = pd.melt(df, id_vars=id_vars, value_vars=date_columns, var_name='Date', value_name='Deaths')

# Convert 'Date' to datetime format
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%m/%d/%y')

# Create new temporal features on df_long
df_long['Day_of_Week'] = df_long['Date'].dt.dayofweek
df_long['Month'] = df_long['Date'].dt.month
df_long['Day'] = df_long['Date'].dt.day
df_long['Days_Since_First_Case'] = (df_long['Date'] - pd.to_datetime('2020-01-22')).dt.days

# Aggregate to find the last death count by state
state_date_death_totals = df_long.groupby(['Province_State', 'Date'])['Deaths'].max().reset_index()

# Find the last valid death count in each row across the date columns
df['Last_Death_Count'] = df[date_columns].apply(lambda row: row[row.last_valid_index()] if not row.isnull().all() else np.nan, axis=1)

# Group by 'Province_State' and sum up the 'Last_Death_Count' to get the total for each state
state_death_totals = df.groupby('Province_State')['Last_Death_Count'].sum().reset_index()

# Save to CSV
state_death_totals.to_csv('state_death_totals.csv', index=False)

# Print or further process the state_death_totals
print('State death totals:', state_death_totals)

# Calculating statistics for deaths
mean_deaths = state_death_totals['Last_Death_Count'].mean()
std_deaths = state_death_totals['Last_Death_Count'].std()
var_deaths = state_death_totals['Last_Death_Count'].var()
# mad_deaths = state_death_totals['Last_Death_Count'].mad()



# # Plotting Deaths with Statistics in Title
# plt.figure(figsize=(14, 8))
# ax = plt.subplot(111)
# bars = ax.bar(state_death_totals['Province_State'], state_death_totals['Last_Death_Count'])
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.3)
# ax.set_xlabel('State')
# ax.set_ylabel('Total Deaths')
# # ax.set_title(f'Total COVID-19 Deaths by State - Mean: {mean_deaths:.2f}, Std: {std_deaths:.2f}, Var: {var_deaths:.2f}, MAD: {mad_deaths:.2f}')
# ax.set_title(f'Total COVID-19 Deaths by State - Mean: {mean_deaths:.2f}, Std: {std_deaths:.2f}, Var: {var_deaths:.2f}')

# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom', rotation=90)
# plt.grid(True)
# plt.show()



######################################################### CONFIRMED DATASET #########################################################

# Load the CONFIRMED dataset
df_confirmed = pd.read_csv('time_series_covid19_confirmed_US.csv')
df_confirmed = df_confirmed.fillna(0)

# Define the columns that uniquely identify each row but aren't part of the time series data
id_vars = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key']

# Columns that are the time series data (i.e., the dates)
date_columns = df_confirmed.columns[df_confirmed.columns.get_loc('1/22/20'):]

# Melt the DataFrame to make 'Date' and 'Cases' columns
df_long_cases = pd.melt(df_confirmed, id_vars=id_vars, value_vars=date_columns, var_name='Date', value_name='Cases')

# Convert 'Date' to datetime format
df_long_cases['Date'] = pd.to_datetime(df_long_cases['Date'], format='%m/%d/%y')

# Create new temporal features on df_long_cases
df_long_cases['Day_of_Week'] = df_long_cases['Date'].dt.dayofweek
df_long_cases['Month'] = df_long_cases['Date'].dt.month
df_long_cases['Day'] = df_long_cases['Date'].dt.day
df_long_cases['Days_Since_First_Case'] = (df_long_cases['Date'] - pd.to_datetime('2020-01-22')).dt.days

# Aggregate to find the last case count by state
state_date_case_totals = df_long_cases.groupby(['Province_State', 'Date'])['Cases'].max().reset_index()

# Find the last valid case count in each row across the date columns
df_confirmed['Last_Case_Count'] = df_confirmed[date_columns].apply(lambda row: row[row.last_valid_index()] if not row.isnull().all() else np.nan, axis=1)

# Group by 'Province_State' and sum up the 'Last_Case_Count' to get the total for each state
state_case_totals = df_confirmed.groupby('Province_State')['Last_Case_Count'].sum().reset_index()

# Save to CSV
state_case_totals.to_csv('state_case_totals.csv', index=False)

# Print or further process the state_case_totals
print('State confirmed totals:', state_case_totals)

# Calculating statistics for cases
mean_cases = state_case_totals['Last_Case_Count'].mean()
std_cases = state_case_totals['Last_Case_Count'].std()
var_cases = state_case_totals['Last_Case_Count'].var()
# mad_cases = state_case_totals['Last_Case_Count'].mad()


# # Plotting Cases with Statistics in Title
# plt.figure(figsize=(14, 8))
# ax = plt.subplot(111)
# bars = ax.bar(state_case_totals['Province_State'], state_case_totals['Last_Case_Count'])
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.3)
# ax.set_xlabel('State')
# ax.set_ylabel('Total Confirms')
# # ax.set_title(f'Total COVID-19 Confirms by State - Mean: {mean_cases:.2f}, Std: {std_cases:.2f}, Var: {var_cases:.2f}, MAD: {mad_cases:.2f}')
# ax.set_title(f'Total COVID-19 Confirms by State - Mean: {mean_cases:.2f}, Std: {std_cases:.2f}, Var: {var_cases:.2f}')

# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), va='bottom', rotation=90)
# plt.grid(True)
# plt.show()


# ######################################################################## FEATURES ######################################################################

# # Example to calculate rolling averages
# state_date_death_totals['Rolling_Avg_Deaths'] = state_date_death_totals.groupby('Province_State')['Deaths'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

# # Calculate rolling averages for cases
# state_date_case_totals['Rolling_Avg_Cases'] = state_date_case_totals.groupby('Province_State')['Cases'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

# # Calculate Case Fatality Rate
# CFR = state_date_death_totals['Rolling_Avg_Deaths'] / state_date_case_totals['Rolling_Avg_Cases']

# print(CFR)

# ######################################################################## TRAINING ######################################################################
# scaler = MinMaxScaler()

# X = scaler.fit_transform(state_case_totals[['Last_Case_Count']])
# y = scaler.fit_transform(state_death_totals[['Last_Death_Count']].values.reshape(-1,1))


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a Sequential model
# model = Sequential([
#     Dense(10, activation='relu', input_shape=(1,)),  # Input layer with 10 neurons
#     Dense(1)  # Output layer with 1 neuron (prediction of death counts)
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mse')

# history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# test_loss = model.evaluate(X_test, y_test)
# print(f"Test loss: {test_loss}")




# ######################################################################## PREDICTION/VISUALISATION ######################################################################


# predictions = model.predict(X_test)


# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('Training vs Validation Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()


# # Assuming 'df' is your DataFrame and it has already been cleaned and prepared
# def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
#     dataframe = dataframe.copy()
#     # Ensuring the target column exists and is correctly named
#     if 'Last_Death_Count' in dataframe:
#         labels = dataframe.pop('Last_Death_Count')
#     else:
#         raise ValueError("DataFrame does not contain the column 'Last_Death_Count'")

#     # Convert all columns to numeric, handling non-numeric gracefully
#     for col in dataframe.columns:
#         dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
#     # Drop NaN values which can create misalignment
#     dataframe.dropna(inplace=True)
#     labels = labels.loc[dataframe.index]  # Ensure labels match the dataframe's rows after dropna

#     # Check that dataframe and labels are aligned
#     if len(dataframe) != len(labels):
#         raise ValueError("The length of the DataFrame and labels do not match after processing.")

#     # Create TensorFlow dataset
#     ds = tf.data.Dataset.from_tensor_slices((dataframe.to_dict('list'), labels.values))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#     return ds


# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # Adjusting to make 0.2 of the original

# train_ds = dataframe_to_dataset(train_df)
# val_ds = dataframe_to_dataset(val_df, shuffle=False)
# test_ds = dataframe_to_dataset(test_df, shuffle=False)

# test_predictions = model.predict(test_ds).flatten()

# # Preparing true values (labels) for comparison
# true_values = tf.concat([y for x, y in test_ds], axis=0)

# # Visualizing the predictions
# plt.figure(figsize=(10, 6))
# plt.scatter(true_values, test_predictions, alpha=0.5)
# plt.xlabel('True Values [Deaths]')
# plt.ylabel('Predictions [Deaths]')
# plt.axis('equal')
# plt.axis('square')
# plt.plot([true_values.numpy().min(), true_values.numpy().max()],
#          [true_values.numpy().min(), true_values.numpy().max()], c='red')
# plt.show()

######################################################################## TENSORFLOW DATASETS ######################################################################


# Function to split and save datasets
def prepare_dataset(csv_input, train_output, val_output, test_output):
    df = pd.read_csv(csv_input)
    # Assuming preprocessing is done here if needed
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # Splits 60% train, 20% validation, 20% test
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)

# Prepare confirmed cases dataset
prepare_dataset('time_series_covid19_confirmed_US.csv', 'Train_cases.csv', 'Validation_cases.csv', 'Test_cases.csv')

# Prepare deaths dataset
prepare_dataset('time_series_covid19_deaths_US.csv', 'Train_deaths.csv', 'Validation_deaths.csv', 'Test_deaths.csv')

class Covid19Deaths(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="COVID-19 Deaths Dataset.",
            features=tfds.features.FeaturesDict({
                "features": tfds.features.Tensor(shape=(10,), dtype=tf.float32),
                "label": tf.int64,
            }),
            supervised_keys=('features', 'label'),
            homepage='https://github.com/CSSEGISandData/COVID-19',
            citation=r"""@article{yourcitation}""",
        )

    def _split_generators(self, dl_manager):
        # You might need to implement downloading here if your data is not locally available
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "filepath": 'Train_deaths.csv'
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "filepath": 'Validation_deaths.csv'
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "filepath": 'Test_deaths.csv'
                }),
        ]



    def _generate_examples(self, filepath):
        # Reading data from the file and yielding examples
        data = pd.read_csv(filepath)
        for index, row in data.iterrows():
            # Assuming all columns except the last are features and the last is the label
            features = row[:-1].to_dict()
            label = row[-1]
            yield index, {
                'features': features,
                'label': label
            }


    
# Assuming your dataset is registered and accessible by TFDS
ds = tfds.load('covid19_deaths', split='train', shuffle_files=True)
for example in ds.take(1):
    print(example)