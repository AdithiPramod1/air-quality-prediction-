import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/aqi_dataset.csv")

# Check data
print(df.head())

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Apply to dataset
df['AQI_Category'] = df['AQI'].apply(get_aqi_category)

def is_high_risk(aqi):
    return aqi > 300

df['High_Risk'] = df['AQI'].apply(is_high_risk)

def get_alert(aqi):
    if aqi > 300:
        return "Severe - Stay Indoors"
    elif aqi > 200:
        return "Poor - Limit Outdoor Activity"
    else:
        return "Safe"

df['Alert'] = df['AQI'].apply(get_alert)

df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')

plt.figure()
plt.plot(df['Datetime'], df['AQI'])
plt.xlabel("Datetime")
plt.ylabel("AQI")
plt.title("AQI Trend")
plt.xticks(rotation=45)
plt.show()

plt.figure()

plt.plot(df['Datetime'], df['AQI'])

high_risk = df[df['High_Risk'] == True]
plt.scatter(high_risk['Datetime'], high_risk['AQI'])

plt.title("AQI with High Risk Days")
plt.show()

df['AQI_MA'] = df['AQI'].rolling(7).mean()

plt.figure()
plt.plot(df['Datetime'], df['AQI'])
plt.plot(df['Datetime'], df['AQI_MA'])

plt.title("AQI Trend with Moving Average")
plt.show()

df.to_csv("data/processed_aqi.csv", index=False)
