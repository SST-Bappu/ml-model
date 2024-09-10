import pandas as pd

# Load the dataset
def load_data():
    df = pd.read_csv('IMDB Dataset.csv')
    return df
    # # Explore the dataset
    # print(df.head())
    # print(df['sentiment'].value_counts())