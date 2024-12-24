import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import aiohttp
import asyncio
from concurrent.futures import ProcessPoolExecutor

def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['timestamp'])

def calculate_moving_average(data, window=30):
    data['moving_avg'] = data['temperature'].rolling(window=window).mean()
    data['moving_std'] = data['temperature'].rolling(window=window).std()
    return data

def detect_anomalies(data):
    data['anomaly'] = (
        (data['temperature'] > data['moving_avg'] + 2 * data['moving_std']) |
        (data['temperature'] < data['moving_avg'] - 2 * data['moving_std'])
    )
    return data

def seasonal_statistics(data):
    return data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

def parallel_analyze(city_data):
    city_data = calculate_moving_average(city_data)
    city_data = detect_anomalies(city_data)
    return city_data

# Параллельный анализ данных в среднем работал на 1 секунду дольше чем последовательный подход
# Связано это, вероятнее всего, со следующим:
# 1) Датасет маленький
# 2) Для создания новых потоков (а точнее процессов, в Python нет настоящей многопоточки из-за GIL) требуются ресурсы (время и память)
# Из-за этих двух факторов последовательный подход обыгрывает параллельный потому что один процесс успевает быстро обработать маленький датасет, 
# в то время, как параллельный подход тратит время на создание новых процессов
# Вывод: параллельный подход проигрывает последовательному из-за маленького датасета и выделения ресурсов на создание новых процессов
def analyze_in_parallel(data):
    cities = data['city'].unique()
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_analyze, [data[data['city'] == city] for city in cities]))
    return pd.concat(results)

# Синхронный подход будет лучше в случае, когда нам нужно запросить данные только для одного города (наш случай)
# Асинхронный подход лучше использовать в случае, когда выбирается несколько городов
# Вывод: не имеет смысла использовать асинхронный подход
async def fetch_temperature(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            elif response.status_code == 401:
                return {"code": 401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}
            else:
                return None

def fetch_temperature_sync(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['main']['temp']
    elif response.status_code == 401:
        return {"code": 401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}
    return None

def main():
    st.title("Temperature Analysis and Monitoring")

    file = st.file_uploader("Upload temperature data (CSV):", type=['csv'])
    if file is not None:
        data = load_data(file)
        st.write("Loaded Data:", data.head())

        city = st.selectbox("Select City:", options=data['city'].unique())
        city_data = data[data['city'] == city]

        st.header("Historical Temperature Analysis")
        # city_data = analyze_in_parallel(city_data)
        city_data = calculate_moving_average(city_data)
        city_data = detect_anomalies(city_data)

        season_stats = seasonal_statistics(data)
        st.write("Seasonal Statistics:", season_stats)

        st.subheader("Temperature Time Series with Anomalies")
        fig, ax = plt.subplots()
        ax.plot(city_data['timestamp'], city_data['temperature'], label='Temperature')
        ax.plot(city_data['timestamp'], city_data['moving_avg'], label='Moving Avg', linestyle='--')
        anomalies = city_data[city_data['anomaly']]
        ax.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Anomalies')
        ax.legend()
        st.pyplot(fig)

        api_key = st.text_input("Enter OpenWeatherMap API Key:")
        if api_key:
            st.header("Current Temperature Monitoring")

            # current_temp = asyncio.run(fetch_temperature_sync(city, api_key))
            current_temp = fetch_temperature_sync(city, api_key)
            if current_temp is not None:
                st.write(f"Current Temperature in {city}: {current_temp}°C")

                season = city_data['season'].iloc[0] 
                stats = season_stats[(season_stats['city'] == city) & (season_stats['season'] == season)]
                mean_temp = stats['mean'].values[0]
                std_temp = stats['std'].values[0]

                lower_bound = mean_temp - 2 * std_temp
                upper_bound = mean_temp + 2 * std_temp

                if lower_bound <= current_temp <= upper_bound:
                    st.write("The current temperature is within the normal range for this season.")
                else:
                    st.write("The current temperature is outside the normal range for this season.")
            else:
                st.write("Failed to fetch current temperature. Check your API key.")

if __name__ == "__main__":
    main()
