import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Any, Dict, List
import requests
import toml
from PIL import Image
from typing import Any, Dict, Tuple

import io
from pathlib import Path


from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

st.set_page_config(
  page_title="Dashboard Prediction 4G LTE",
  page_icon="🧊",
  layout="wide",
  initial_sidebar_state="expanded",
  menu_items={
     'Get Help': 'https://www.extremelycoolapp.com/help',
     'Report a bug': "https://www.extremelycoolapp.com/bug",
     'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def get_project_root() -> str:

  return str(Path(__file__).parent)

# Load TOML config file

@st.cache(allow_output_mutation=True, ttl=300)
def load_config(
    config_readme_filename: str
    
) -> Dict[Any, Any]:

  config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
  return dict(config_readme)
# @st.cache(ttl=300)
# def load_image(image_name: str) -> Image:
#     """Displays an image.

#     Parameters
#     ----------
#     image_name : str
#         Local path of the image.

#     Returns
#     -------
#     Image
#         Image to be displayed.
#     """
#     return Image.open(Path(get_project_root()) / f"references/{image_name}")

readme = load_config("config_readme.toml") #Fungsi untuk me-load file untuk fitur guide user

st.markdown("""
                    <h1 style='text-align: center;'>\
                        Main Dashboard 4G LTE</h1>
                    """, 
                    unsafe_allow_html=True)
st.write("")
with st.expander("Apa itu Main Dashboard 4G LTE", expanded=False):
    st.write(readme["app"]["app_definition"])
with st.expander("Apa saja bagian utama dari dashboard ini?", expanded=False):
    st.write(readme["app"]["app_rules"])

# 1. Dataset
image = Image.open('4G.png')
st.sidebar.image(image)
st.sidebar.title("1. Loading Data")
st.sidebar.write("Pengguna bisa memilih salah satu opsi file dataset untuk pengukuran throughput")
data = None

if st.sidebar.checkbox("Load Existing Dataset", value = False,  help=readme["tooltips"]["existing_upload"]) == True: 
    filename = st.sidebar.selectbox('Choose one files',  
                            ('','S9-9am-20191124.csv', 'S9-12pm-20191124.csv', 
                            'S9-6pm-20191124.csv', 'S10e-9am-20191124.csv', 
                            'S10e-12pm-20191124.csv', 'S10e-6pm-20191124.csv'),
                            help=readme["tooltips"]["select_files"])
    if filename:
        data = pd.read_csv(filename, encoding='utf-7')
        data = data[['Timestamp', 'Longitude', 'Latitude', 'Speed', 'Operator', 'CellID', 
        'LAC', 'LTERSSI', 'RSRP', 'RSRQ', 'SNR', 'DL_bitrate', 'UL_bitrate']] #dataframe untuk menyimpan setiap data dalam column       
else:
    file = st.sidebar.file_uploader('Upload file', type=['csv', 'txt'], help=readme["tooltips"]["data_upload"])
    if file is not None:
        # if file.type != 'text/csv' and file.type != 'text/plain':
        # #     st.warning("Format File beserta isinya tidak mendukung")
        if file.type == 'text/csv' or file.type == 'application/vnd.ms-excel':
            data = pd.read_csv(file, encoding='utf-7')
            data = data[['Timestamp', 'Longitude', 'Latitude', 'Speed', 'Operator', 'CellID', 
            'LAC', 'LTERSSI', 'RSRP', 'RSRQ', 'SNR', 'DL_bitrate', 'UL_bitrate']]
        elif file.type == 'text/plain':
            data = pd.read_csv(file, delimiter=r"\s+")
            data = data.rename(columns={
                        'Level' : 'RSRP',
                        'Qual' : 'RSRQ'})
            data = data[['Timestamp', 'Longitude', 'Latitude', 'Speed', 'Operator', 'CellID', 
            'LAC', 'LTERSSI', 'RSRP', 'RSRQ', 'SNR', 'DL_bitrate', 'UL_bitrate']]
            data['Timestamp'] = data['Timestamp'].apply(lambda x: x.split('_')[0].replace('.', '-') + ' ' + x.split('_')[1].replace('.', ':')) #transformasi bentuk timestamp menjadi format yang disupport oleh pandas
            data['SNR'] = pd.to_numeric(data['SNR'],errors='coerce')
            data['RSRP'] = pd.to_numeric(data['RSRP'],errors='coerce')
            data['LTERSSI'] = pd.to_numeric(data['LTERSSI'],errors='coerce')
            data['UL_bitrate'] = pd.to_numeric(data['UL_bitrate'],errors='coerce')
            data['RSRQ'] = pd.to_numeric(data['RSRQ'],errors='coerce')
            data['CellID'] = pd.to_numeric(data['CellID'],errors='coerce')
            data['LAC'] = pd.to_numeric(data['LAC'],errors='coerce') 

if data is not None:
    st.markdown(""" """)
    st.sidebar.title("2. Feature Selection")
    st.sidebar.write("Silahkan pilih beberapa fitur yang dapat memodelkan data")
    heatmap = st.sidebar.checkbox("Launch Heatmap", value=False, help=readme["tooltips"]["show_heatmap"])
    st.write(help=readme["app"]["app_correlation"])
    figure = plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr().round(2)
    sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    if heatmap == True:
        st.subheader("Correlation Heatmap")
        st.pyplot(figure)
        with st.expander("Apa yang wajib pengguna amati dari informasi Correlation Heatmap", expanded=False):
            st.write(readme["app"]["app_correlation"])
            st.subheader('')

    

    # 2. Data Preparation   
    option = list(data.columns)[1:]
    option.remove('DL_bitrate')
    features_names = st.sidebar.multiselect("Choose Features", option, help=readme["tooltips"]["select_feature"]) 

    target_names = 'DL_bitrate'
    features = data[['Timestamp'] + features_names + [target_names]] # dataframe dengan column yang dipilih sebagai feature
    # target = data[target] # dataframe dengan column target yang dipilih (by default isinya cuma DL_bitrate)

    if len(features_names) >= 3:
        if st.sidebar.checkbox("PCA (Optional)", help=readme["tooltips"]["cekbox_pca"]):
            # pca_features = st.sidebar.multiselect("Pilih fitur mana yang akan diproses kedalam PCA", features_names, default=features_names, help=readme["tooltips"]["select_pca"]) # pilih variabel/column2 yang akan digabungkan menggunakan pca
            pca_features = features_names
            # PCA ditambah disini kecuali Timestamp
            
            if len(pca_features):
                pca = PCA(n_components=1, random_state=123)
                pca.fit(features[pca_features])
                features['performance'] = pca.transform(features.loc[:, pca_features]).flatten()
                features.drop(pca_features, axis=1, inplace=True)

    # Processing datetime
    if 'Timestamp' in features:
        timestamp = pd.to_datetime(features['Timestamp'])
        features['Timestamp'] = timestamp
        features['day'] = timestamp.dt.day
        features['month'] = timestamp.dt.month
        features['year'] = timestamp.dt.year
        features['hour'] = timestamp.dt.hour
        features['minute'] = timestamp.dt.minute
        features['second'] = timestamp.dt.second
    
    target = features[target_names]
    features = features.drop(target_names,axis =1)
    # 3. Data Modelling
    st.sidebar.title("3. Modelling and Evaluation")
    st.sidebar.write("Bagian ini digunakan untuk proses modelling and data visualization")
    model = st.sidebar.selectbox("Choose model", options=[' ','k-Nearest Neighbor', 'Random Forest', 'Boosting Algorithm'], help=readme["tooltips"]["regressor"])
    trainsize = st.sidebar.slider("Select train size", min_value=0.01, max_value=0.99, value=0.5, help=readme["tooltips"]["train_test_split"])
    
    cutoff = int(len(data)*trainsize)

    train_features, test_features = features[:cutoff], features[cutoff:]
    train_target, test_target = target[:cutoff], target[cutoff:]

    if model == 'k-Nearest Neighbor':
        n_neighbor = st.sidebar.slider('Number Neighbor', 1, 20, value=5, help=readme["tooltips"]["knn"])

        knn = KNeighborsRegressor(n_neighbors=n_neighbor)
        knn.fit(train_features.drop('Timestamp', axis=1) if 'Timestamp' in train_features.columns else train_features, train_target)


        pred_target = knn.predict(test_features.drop('Timestamp', axis=1) if 'Timestamp' in train_features.columns else train_features)
        test_target = np.array(test_target)

        #visualisasi
        # visualization = pd.DataFrame(data={'Actual': test_target, 'Predicted': pred_target}, index=pd.DatetimeIndex(test_features['Timestamp']))
        visualization = pd.DataFrame(data={'Actual': test_target, 'Predicted': pred_target}, index=range(len(test_target)))
        if st.checkbox('Launch Visualization', help=readme["app"]["show_visual"]):
            st.write('')
            st.subheader("Throughput vs Time")
            st.line_chart(visualization)
            with st.expander("Info Seputar Grafik Throughput vs Time", expanded=False):
                st.write(readme["app"]["app_visualization"])

            #metrics
            mae = metrics.mean_absolute_error(test_target, pred_target)
            mse = metrics.mean_squared_error(test_target, pred_target)
            rmse = np.sqrt(metrics.mean_squared_error(test_target, pred_target))
            r2_square = metrics.r2_score(test_target, pred_target)
            st.subheader("Evaluation Metric Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R2 Score",  "{:.2f}".format(r2_square)) 
            col2.metric("RMSE",  "{:.2f}".format(rmse))
            col3.metric("MSE", "{:.2f}".format(mse))
            col4.metric("MAE", "{:.2f}".format(mae))
            with st.expander("Informasi Seputar Evaluation Performance", expanded=False):
                st.write(readme["app"]["app_performance"])
                st.write("")
                if st.checkbox("Tampilkan formula metric performance", value=False):
                    st.write("Jika N diketahui merupakan banyaknya baris dari dataset train pada sekumpulan pengujian maka:")
                    st.latex(r"R^2 SCORE = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Forecast - Forecast_i)^2}")
                    st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                    st.latex(r"MSE = \dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2")
                    st.latex(r"MAE = \dfrac{1}{N}\sum_{i=1}^{N}|Truth_i - Forecast_i|")
            with st.expander("Kriteria Penting seputar Metric Performance?", expanded=False):
                st.write(readme["app"]["metric_criteria"])
                st.write("")

    elif model == 'Random Forest':
        n_estimators = st.sidebar.slider('Number Estimator', 1, 1000, value=100, help=readme["tooltips"]["random_forest"])

        RF = RandomForestRegressor(n_estimators=n_estimators)
        RF.fit(train_features.drop('Timestamp', axis=1) if 'Timestamp' in train_features.columns else train_features, train_target)

        pred_target = RF.predict(test_features.drop('Timestamp', axis=1))
        test_target = np.array(test_target)

        #visualisasi
        # visualization = pd.DataFrame(data={'Actual': test_target, 'Predicted': pred_target}, index=pd.DatetimeIndex(test_features['Timestamp']))
        visualization = pd.DataFrame(data={'Actual': test_target, 'Predicted': pred_target}, index=range(len(test_target)))
        if st.checkbox('Launch Visualization', help=readme["app"]["show_visual"]):
            st.write('')
            st.subheader("Throughput vs Time")
            st.line_chart(visualization)
            with st.expander("Info Seputar Grafik Throughput vs Time", expanded=False):
                            st.write(readme["app"]["app_visualization"])
        #metrics
            mae = metrics.mean_absolute_error(test_target, pred_target)
            mse = metrics.mean_squared_error(test_target, pred_target)
            rmse = np.sqrt(metrics.mean_squared_error(test_target, pred_target))
            r2_square = metrics.r2_score(test_target, pred_target)
            st.subheader("Evaluation Metric Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R2 Score",  "{:.2f}".format(r2_square)) 
            col2.metric("RMSE",  "{:.2f}".format(rmse))
            col3.metric("MSE", "{:.2f}".format(mse))
            col4.metric("MAE", "{:.2f}".format(mae))
            with st.expander("Informasi Seputar Evaluation Performance", expanded=False):
                st.write(readme["app"]["app_performance"])
                st.write("")
                if st.checkbox("Tampilkan formula metric performance", value=False):
                    st.write("Jika N diketahui merupakan banyaknya baris dari dataset train pada sekumpulan pengujian maka:")
                    st.latex(r"R^2 SCORE = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Forecast - Forecast_i)^2}")
                    st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                    st.latex(r"MSE = \dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2")
                    st.latex(r"MAE = \dfrac{1}{N}\sum_{i=1}^{N}|Truth_i - Forecast_i|")
            with st.expander("Kriteria Penting seputar Metric Performance?", expanded=False):
                st.write(readme["app"]["metric_criteria"])
                st.write("")


    elif model == 'Boosting Algorithm':
        n_estimators = st.sidebar.slider('Number Estimator', 1, 1000, value=10, help=readme["tooltips"]["ada_boost"])

        boosting = AdaBoostRegressor(n_estimators=n_estimators)
        boosting.fit(train_features.drop('Timestamp', axis=1) if 'Timestamp' in train_features.columns else train_features, train_target)

        pred_target = boosting.predict(test_features.drop('Timestamp', axis=1))
        test_target = np.array(test_target)

        #visualisasi
        # visualization = pd.DataFrame(data={'Actual': test_target, 'Predicted': pred_target}, index=pd.DatetimeIndex(test_features['Timestamp']))
        visualization = pd.DataFrame(data={'Actual': test_target, 'Predicted': pred_target}, index=range(len(test_target)))
        if st.checkbox('Launch Visualization', help=readme["app"]["show_visual"]):
            st.write('')
            st.subheader("Throughput vs Time")
            st.line_chart(visualization)
            with st.expander("Info Seputar Grafik Throughput vs Time", expanded=False):
                            st.write(readme["app"]["app_visualization"])

            #metrics
            mae = metrics.mean_absolute_error(test_target, pred_target)
            mse = metrics.mean_squared_error(test_target, pred_target)
            rmse = np.sqrt(metrics.mean_squared_error(test_target, pred_target))
            r2_square = metrics.r2_score(test_target, pred_target)
            st.subheader("Evaluation Metric Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R2 Score",  "{:.2f}".format(r2_square)) 
            col2.metric("RMSE",  "{:.2f}".format(rmse))
            col3.metric("MSE", "{:.2f}".format(mse))
            col4.metric("MAE", "{:.2f}".format(mae))
            with st.expander("Informasi seputar evaluation performance?", expanded=False):
                st.write(readme["app"]["app_performance"])
                st.write("")
                if st.checkbox("Tampilkan formula metric performance", value=False):
                    st.write("Jika N diketahui merupakan banyaknya baris dari dataset train pada sekumpulan pengujian maka:")
                    st.latex(r"R^2 SCORE = 1 - \dfrac{\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}{\sum_{i=1}^{N}(Forecast - Forecast_i)^2}")
                    st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2}")
                    st.latex(r"MSE = \dfrac{1}{N}\sum_{i=1}^{N}(Truth_i - Forecast_i)^2")
                    st.latex(r"MAE = \dfrac{1}{N}\sum_{i=1}^{N}|Truth_i - Forecast_i|")
            with st.expander("Kriteria Penting seputar Metric Performance?", expanded=False):
                st.write(readme["app"]["metric_criteria"])
                st.write("")

st.sidebar.caption("Copyright © 2021-2022")





