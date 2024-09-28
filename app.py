import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

st.set_page_config(layout="wide")

df=pd.read_csv('24-hours dataset.csv')

df['timestamp']=pd.to_datetime(df['timestamp'])

st.title("Graph Implementation")
    
st.write("### Data Preview")
st.dataframe(df.head())

def assign_compensation_method(row):
    if row['is_holiday'] == 1:
        return 'Lower Power Supply'
    elif row['solar_generation'] > 200:
        return 'Increased Renewable Energy Integration'
    elif row['hour_of_day'] >= 18 and row['hour_of_day'] <= 22:
        return 'Peak Load Shifting'
    elif row['temperature'] > 35:
        return 'Increased Power Supply'
    else:
        return 'Normal Operation'

df['compensation_method'] = df.apply(assign_compensation_method, axis=1)

print(df[['timestamp', 'load', 'compensation_method']].head(10))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = df[['temperature', 'solar_generation', 'hour_of_day', 'load', 'is_holiday']]
y=df['compensation_method']
X_reg = df[['temperature', 'solar_generation', 'hour_of_day', 'is_holiday']]
y_reg = df['load']
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_clf=RandomForestClassifier(n_estimators=100, random_state=42)

rf_reg.fit(X_reg_train, y_reg_train)

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
class_accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Model Accuracy: {class_accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Make predictions on the test set for regression (peak supply)
y_reg_pred = rf_reg.predict(X_reg_test)
reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Regression Model Mean Squared Error: {reg_mse:.2f}")
print(confusion_matrix(y_test, y_pred))

def predict_for_date(date, temperature, solar_generation, hour_of_day, is_holiday, rf_clf, rf_reg, le):
    new_data_class = pd.DataFrame({
        'temperature': [temperature],
        'solar_generation': [solar_generation],
        'hour_of_day': [hour_of_day],
        'load': [0],
        'is_holiday': [is_holiday]
    })

    new_data_reg = pd.DataFrame({
        'temperature': [temperature],
        'solar_generation': [solar_generation],
        'hour_of_day': [hour_of_day],
        'is_holiday': [is_holiday]
    })
    
    predicted_class_label = rf_clf.predict(new_data_class)
    predicted_compensation_method = le.inverse_transform(predicted_class_label)
    predicted_peak_supply = rf_reg.predict(new_data_reg)

    return {
        'date': date,
        'hour_of_day': hour_of_day,
        'compensation_method': predicted_compensation_method[0],
        'peak_supply_needed': predicted_peak_supply[0]
    }

def generate_24_hour_predictions(timestamp, temperature, solar_generation, is_holiday, rf_clf, rf_reg, le):

    results = []

    hour=timestamp.hour
    
    prediction = predict_for_date(timestamp, temperature, solar_generation, hour, is_holiday, rf_clf, rf_reg, le)
    results.append(prediction)
    
    df_predictions = pd.DataFrame(results)
    
    return df_predictions


testdf=pd.read_csv('24-hours testset.csv')

testdf.drop('electricity_demand', axis=1, inplace=True)

testdf['timestamp']=pd.to_datetime(testdf['timestamp'])

def generate_year_predictions(test_dataset, rf_clf, rf_reg, le):
    
    all_predictions = []

    for _, row in test_dataset.iterrows():
        timestamp = pd.to_datetime(row['timestamp'])
        temperature = row['temperature']
        solar_generation = row['solar_generation']
        is_holiday = row['is_holiday']

        # Generate predictions for each hour of the day
        daily_predictions = generate_24_hour_predictions(timestamp, temperature, solar_generation, is_holiday, rf_clf, rf_reg, le)
        all_predictions.append(daily_predictions)
    
    # Combine all daily predictions into one DataFrame
    df_all_predictions = pd.concat(all_predictions, ignore_index=True)
    
    return df_all_predictions

df_year_predictions = generate_year_predictions(testdf, rf_clf, rf_reg, le)

df_year_predictions

def plot_predictions(df_predictions):

    fig= plt.subplots()
    
    plt.figure(figsize=(14, 7))

    sns.lineplot(data=df_year_predictions, x='hour_of_day', y='peak_supply_needed', marker='o', color='blue', label='Peak Supply Needed (MW)')
    
    sns.lineplot(data=df.head(100), x='hour_of_day', y='load', palette='Set1')

    plt.xlabel('Hour of the Day')
    plt.ylabel('Peak Supply Needed (MW)')
    plt.legend(title='Compensation Method')
    plt.grid(True)
    # plt.show()

    st.pyplot(plt)

plot_predictions(df_year_predictions)

full_testdf['old_compensation']=df_year_predictions['compensation_method']

predicted_loads = df_year_predictions['peak_supply_needed'].head(8737)
actual_loads = df['load']

mae = mean_absolute_error(actual_loads, predicted_loads)
mse = mean_squared_error(actual_loads, predicted_loads)
rmse = np.sqrt(mse)
r2 = r2_score(actual_loads, predicted_loads)
mape = np.mean(np.abs((actual_loads - predicted_loads) / actual_loads)) * 100
accuracy = 100 - mape

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"Accuracy: {accuracy:.4f}%")