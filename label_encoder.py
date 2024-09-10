from sklearn.preprocessing import LabelEncoder
from load_data import load_data

def label_encoder():
    # Initialize the label encoder
    le = LabelEncoder()
    df = load_data()
    # Transform the sentiment labels into numeric values (0: negative, 1: positive)
    y = le.fit_transform(df['sentiment'])
    
    return y

if __name__ == '__main__':
    label_encoder()