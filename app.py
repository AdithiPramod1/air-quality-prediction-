import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------

df = pd.read_csv("data/aqi_dataset.csv")

# Fix Datetime → Date
df.rename(columns={'Datetime': 'Date'}, inplace=True)

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort data
df = df.sort_values('Date')

# Load model
model = joblib.load("aqi_model.pkl")

# Features used in model
features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Month','DayOfWeek']

# ---------------- AQI CATEGORY ----------------

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

# ---------------- UI ----------------

st.title(" Air Quality Index Dashboard")

# Sidebar
city = st.sidebar.selectbox("Select City", df['City'].unique())

# Filter data
city_df = df[df['City'] == city].copy()

# Clean AQI
city_df['AQI'] = pd.to_numeric(city_df['AQI'], errors='coerce')
city_df = city_df.dropna(subset=['AQI'])

# Latest AQI
latest = city_df.iloc[-1]
aqi_value = latest['AQI']

# AQI display
st.metric("Current AQI", int(aqi_value), get_aqi_category(aqi_value))

# Alerts
if aqi_value > 300:
    st.error(" Severe Air Quality! Stay Indoors")
elif aqi_value > 200:
    st.warning(" Poor Air Quality")
else:
    st.success("Air Quality is Acceptable")

# ---------------- GRAPH ----------------

st.subheader(" AQI Trend")

fig, ax = plt.subplots()

ax.plot(city_df['Date'], city_df['AQI'])
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.set_title("AQI Trend")

st.pyplot(fig)

# ---------------- PREDICTION ----------------

st.subheader("Predict AQI")

input_data = []
for feature in features:
    val = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(val)

if st.button("Predict AQI"):
    try:
        prediction = model.predict([input_data])[0]
        st.success(f"Predicted AQI: {int(prediction)}")
        st.info(get_aqi_category(prediction))
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------- ADMIN PANEL ----------------

st.subheader(" Admin Panel")

uploaded_file = st.file_uploader("Upload New Dataset", type=["csv"])

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(new_df.head())

    # Required columns
    required_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Month','DayOfWeek','AQI']

    if all(col in new_df.columns for col in required_cols):

        st.success(" Dataset is valid")

        if st.button(" Retrain Model"):

            X = new_df[required_cols[:-1]]
            y = new_df['AQI']

            try:
                model.fit(X, y)

                joblib.dump(model, "aqi_model.pkl")

                st.success(" Model retrained and saved successfully!")

                # Download updated model
                with open("aqi_model.pkl", "rb") as f:
                    st.download_button(
                        label="⬇ Download Updated Model",
                        data=f,
                        file_name="aqi_model.pkl"
                    )

            except Exception as e:
                st.error(f"Error during training: {e}")

    else:
        st.error(" Dataset missing required columns")