from rest_framework.views import APIView
from rest_framework.response import Response
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class Predictor(APIView):
    def post(self, request):
        # Load the trained model
        model = joblib.load('models/model.pkl')
        tfidf = joblib.load('models/tfidf.pkl')
        
        # Get the text from the request
        text = request.data['text']
        
        # Transform the text using the TF-IDF vectorizer
        text_transformed = tfidf.transform([text])
        
        # Make a prediction
        prediction = model.predict(text_transformed)
        # Return the prediction
        return Response({'prediction': 'Positive' if prediction[0] else 'Negative'})