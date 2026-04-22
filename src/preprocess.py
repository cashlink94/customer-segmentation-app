import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # drop ID column if exists
    if "CUST_ID" in df.columns:
        df = df.drop("CUST_ID", axis=1)

    # handle missing values
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed


def scale_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler