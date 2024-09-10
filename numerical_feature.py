from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocess import preprocess_data
# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

def numerical_feature():
    # load cleaned reviews
    df = preprocess_data()
    # Fit and transform the cleaned reviews
    X = tfidf.fit_transform(df['cleaned_review']).toarray()
    
    # Check the shape of the transformed data
    # print(X.shape)
    
    return X, tfidf
if __name__ == '__main__':
    numerical_feature()