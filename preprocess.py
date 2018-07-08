from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import preprocessing

normalized_losses_scaler = MinMaxScaler()
normalized_losses_scaler.fit(np.array([65, 256]).reshape(-1, 1))  # The min max values in know for data description

wheel_base_scaler = MinMaxScaler()
wheel_base_scaler.fit(np.array([86.6, 120.9]).reshape(-1, 1))  # The min max values in know for data description

length_scaler = MinMaxScaler()
length_scaler.fit(np.array([141.1, 208.1]).reshape(-1, 1))  # The min max values in know for data description

width_scaler = MinMaxScaler()
width_scaler.fit(np.array([60.3, 72.3]).reshape(-1, 1))  # The min max values in know for data description

height_scaler = MinMaxScaler()
height_scaler.fit(np.array([47.8, 59.8]).reshape(-1, 1))  # The min max values in know for data description

curb_weight_scaler = MinMaxScaler()
curb_weight_scaler.fit(np.array([1488, 4066]).reshape(-1, 1))  # The min max values in know for data description

engine_size_scaler = MinMaxScaler()
engine_size_scaler.fit(np.array([61, 326]).reshape(-1, 1))  # The min max values in know for data description

bore_scaler = MinMaxScaler()
bore_scaler.fit(np.array([2.54, 3.94]).reshape(-1, 1))  # The min max values in know for data description

stroke_scaler = MinMaxScaler()
stroke_scaler.fit(np.array([2.07, 4.17]).reshape(-1, 1))  # The min max values in know for data description

compression_ratio_scaler = MinMaxScaler()
compression_ratio_scaler.fit(np.array([7, 23]).reshape(-1, 1))  # The min max values in know for data description

horsepower_scaler = MinMaxScaler()
horsepower_scaler.fit(np.array([48, 288]).reshape(-1, 1))  # The min max values in know for data description

peak_rpm_scaler = MinMaxScaler()
peak_rpm_scaler.fit(np.array([4150, 6600]).reshape(-1, 1))  # The min max values in know for data description

city_mpg_scaler = MinMaxScaler()
city_mpg_scaler.fit(np.array([13, 49]).reshape(-1, 1))  # The min max values in know for data description

highway_mpg_scaler = MinMaxScaler()
highway_mpg_scaler.fit(np.array([16, 54]).reshape(-1, 1))  # The min max values in know for data description

price_scaler = MinMaxScaler()
price_scaler.fit(np.array([5118, 45400]).reshape(-1, 1))  # The min max values in know for data description

mark_label_encoder = preprocessing.LabelEncoder()
mark_label_encoder.fit(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
                        'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
                        'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
                        'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'])

fuel_type_label_encoder = preprocessing.LabelEncoder()
fuel_type_label_encoder.fit(['diesel', 'gas'])

aspiration_label_encoder = preprocessing.LabelEncoder()
aspiration_label_encoder.fit(['std', 'turbo'])

num_of_doors_label_encoder = preprocessing.LabelEncoder()
num_of_doors_label_encoder.fit(['four', 'two'])

body_style_label_encoder = preprocessing.LabelEncoder()
body_style_label_encoder.fit(['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible'])

drive_wheels_label_encoder = preprocessing.LabelEncoder()
drive_wheels_label_encoder.fit(['4wd', 'fwd', 'rwd'])

engine_location_label_encoder = preprocessing.LabelEncoder()
engine_location_label_encoder.fit(['front', 'rear'])

engine_type_label_encoder = preprocessing.LabelEncoder()
engine_type_label_encoder.fit(['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'])

num_of_cylinders_label_encoder = preprocessing.LabelEncoder()
num_of_cylinders_label_encoder.fit(['eight', 'five', 'four', 'six', 'three', 'twelve', 'two'])

fuel_system_label_encoder = preprocessing.LabelEncoder()
fuel_system_label_encoder.fit(['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi'])


def preprocess(series):
    series = preprocess_handle_empty_values(series)
    preprocess_scale(series)
    preprocess_make_field(series)
    preprocess_fuel_type_field(series)
    preprocess_aspiration_field(series)
    preprocess_num_of_doors_field(series)
    preprocess_body_style_field(series)
    preprocess_drive_wheels_field(series)
    preprocess_engine_location_field(series)
    preprocess_engine_type_field(series)
    preprocess_num_of_cylinders_field(series)
    preprocess_fuel_system_field(series)
    return series


def preprocess_handle_empty_values(series):
    return series.replace('?', np.nan)


def preprocess_scale(series):
    if not np.isnan(float(series['normalized-losses'])):
        series['normalized-losses'] = normalized_losses_scaler.transform(series['normalized-losses'])[0, 0]
    if not np.isnan(float(series['wheel-base'])):
        series['wheel-base'] = wheel_base_scaler.transform(series['wheel-base'])[0, 0]
    if not np.isnan(float(series['length'])):
        series['length'] = length_scaler.transform(series['length'])[0, 0]
    if not np.isnan(float(series['width'])):
        series['width'] = width_scaler.transform(series['width'])[0, 0]
    if not np.isnan(float(series['height'])):
        series['height'] = height_scaler.transform(series['height'])[0, 0]
    if not np.isnan(float(series['curb-weight'])):
        series['curb-weight'] = curb_weight_scaler.transform(series['curb-weight'])[0, 0]
    if not np.isnan(float(series['engine-size'])):
        series['engine-size'] = engine_size_scaler.transform(series['engine-size'])[0, 0]
    if not np.isnan(float(series['bore'])):
        series['bore'] = bore_scaler.transform(series['bore'])[0, 0]
    if not np.isnan(float(series['stroke'])):
        series['stroke'] = stroke_scaler.transform(series['stroke'])[0, 0]
    if not np.isnan(float(series['compression-ratio'])):
        series['compression-ratio'] = compression_ratio_scaler.transform(series['compression-ratio'])[0, 0]
    if not np.isnan(float(series['horsepower'])):
        series['horsepower'] = horsepower_scaler.transform(series['horsepower'])[0, 0]
    if not np.isnan(float(series['peak-rpm'])):
        series['peak-rpm'] = peak_rpm_scaler.transform(series['peak-rpm'])[0, 0]
    if not np.isnan(float(series['city-mpg'])):
        series['city-mpg'] = city_mpg_scaler.transform(series['city-mpg'])[0, 0]
    if not np.isnan(float(series['highway-mpg'])):
        series['highway-mpg'] = highway_mpg_scaler.transform(series['highway-mpg'])[0, 0]
    if not np.isnan(float(series['price'])):
        series['price'] = price_scaler.transform(series['price'])[0, 0]


def preprocess_wheel_base(series):
    series['wheel-base'] = wheel_base_scaler.transform(series['wheel-base'])[0, 0]


def preprocess_make_field(series):
    series['make'] = mark_label_encoder.transform([series['make']])[0]


def preprocess_fuel_type_field(series):
    series['fuel-type'] = fuel_type_label_encoder.transform([series['fuel-type']])[0]


def preprocess_aspiration_field(series):
    series['aspiration'] = aspiration_label_encoder.transform([series['aspiration']])[0]


def preprocess_num_of_doors_field(series):
    if isinstance(series['num-of-doors'], str):
        series['num-of-doors'] = num_of_doors_label_encoder.transform([series['num-of-doors']])[0]


def preprocess_body_style_field(series):
    series['body-style'] = body_style_label_encoder.transform([series['body-style']])[0]


def preprocess_drive_wheels_field(series):
    series['drive-wheels'] = drive_wheels_label_encoder.transform([series['drive-wheels']])[0]


def preprocess_engine_location_field(series):
    series['engine-location'] = engine_location_label_encoder.transform([series['engine-location']])[0]


def preprocess_engine_type_field(series):
    series['engine-type'] = engine_type_label_encoder.transform([series['engine-type']])[0]


def preprocess_num_of_cylinders_field(series):
    series['num-of-cylinders'] = num_of_cylinders_label_encoder.transform([series['num-of-cylinders']])[0]


def preprocess_fuel_system_field(series):
    series['fuel-system'] = fuel_system_label_encoder.transform([series['fuel-system']])[0]
