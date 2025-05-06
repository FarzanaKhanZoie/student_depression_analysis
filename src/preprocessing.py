import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    imputer = KNNImputer(n_neighbors=1)
    df_imputed = imputer.fit_transform(df[['Financial Stress']])
    df['Financial Stress'] = df_imputed

    encodable_cat_features = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for feature in encodable_cat_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    diet_mapping = {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2, 'Others': 3}
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_mapping)

    sleep_duration_mapping = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3, 'Others': 4}
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_duration_mapping)
    
    drop_fe = ['Gender', 'Work Pressure', 'Job Satisfaction', 'City', 'Degree', 'id', 'Profession', 'CGPA']
    df = df.drop(drop_fe, axis=1)
    
    numerical_features = ['Dietary Habits', 'Sleep Duration', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df
