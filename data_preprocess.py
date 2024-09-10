import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from load_data import load_data

nltk.data.path.append('/Users/sst-bappu/nltk_data')


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords, and lemmatize the words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

def preprocess_data():
    # Load the dataset
    df = load_data()
    
    # Apply the preprocessing function to the reviews
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Check a sample of the cleaned reviews
    # print(df['cleaned_review'].head())
    return df

if __name__ == '__main__':
    preprocess_data()