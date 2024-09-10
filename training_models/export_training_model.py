import joblib
from .train_ml_model import train_ml_model
from data import numerical_feature

def export_ml_model():
    # Export the trained model to a file
    model = train_ml_model()
    X, tfidf = numerical_feature()
    joblib.dump(model, 'models/model.pkl')
    print('Model exported successfully!')
    joblib.dump(tfidf, 'models/tfidf.pkl')
    print('TF-IDF exported successfully!')
    
    return True


if __name__ == '__main__':
    export_ml_model()
    