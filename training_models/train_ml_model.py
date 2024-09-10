from sklearn.linear_model import LogisticRegression
from data import datasets

def train_ml_model():
    X_train, X_test, y_train, y_test = datasets()
    
    # Initialize and train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    return model

# if __name__ == '__main__':
#     train_ml_model()
