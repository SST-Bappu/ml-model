from sklearn.model_selection import train_test_split
from .numerical_feature import numerical_feature
from .label_encoder import label_encoder

def datasets():
    X, tfidf = numerical_feature()
    y = label_encoder()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'Training data: {X_train.shape}, Testing data: {X_test.shape}')

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    datasets()