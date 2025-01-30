import os
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import toml

# NLTK data path configuration
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

# Download and store NLTK data locally
def download_nltk_data():
    """Download required NLTK data to local directory"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', download_dir=NLTK_DATA_PATH)

    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

# Call the download function
download_nltk_data()

# Load API key from config.toml
def load_api_key():
    config = toml.load('.streamlit/config.toml')
    return config['general']['api_key']

# Linking to Google Books API
def search_books(passage):
    api_key = load_api_key()
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        'q': passage,
        'key': api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        books = response.json()
        return books
    else:
        st.error(f"Error: {response.status_code}")
        return None
    
# Counts total number of words in the passage
def count_words(passage):
    words = word_tokenize(passage)
    return len([word for word in words if word not in string.punctuation])

# Counts total number of words without considering stopwords
def count_words_without_stopwords(passage):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(passage)
    return len([word for word in words if word.lower() not in stop_words and word not in string.punctuation])

# Emotion analysis based on compound score
def analyze_emotion(passage):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(passage)
    
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'joy'
    elif compound <= -0.05:
        return 'sadness'
    elif scores['pos'] > scores['neg']:
        return 'surprise'
    elif scores['neg'] > scores['pos']:
        return 'anger'
    elif scores['neg'] > 0.5:  # Threshold for disgust
        return 'disgust'
    elif scores['pos'] < 0.1 and scores['neg'] > 0.1:  # Threshold for fear
        return 'fear'
    else:
        return 'neutral'
    
# Summarizing the passage using LSA
def summarize_with_lsa(passage):
    parser = PlaintextParser.from_string(passage, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join(str(sentence) for sentence in summary)

# Main function to analyze the text and display results
def analyze_text(passage):
    """Main function to analyze the text and return results"""
    results = []
    
    # 1. Word Count
    total_words = count_words(passage)
    total_words_no_stopwords = count_words_without_stopwords(passage)

    results.append(f"Total number of words: {total_words}")
    results.append(f"Total number of words (without stopwords): {total_words_no_stopwords}")
    
    # 2. Emotional Analysis
    emotion = analyze_emotion(passage)
    results.append(f"Predominant emotion: {emotion}\n")

    # 3. Book Search
    results.append("Possible books the passage might be from:")
    books = search_books(passage)
    if books:
        for item in books.get('items', [])[:3]:  # To get the first 3 possible books
            title = item['volumeInfo'].get('title', 'No title found')
            authors = item['volumeInfo'].get('authors', ['No authors found'])
            results.append(f"- Title: {title}")
            results.append(f"  Authors: {', '.join(authors)}\n")
    
    # 4. Summary using LSA
    lsa_summary = summarize_with_lsa(passage)
    results.append(f"Summary:\n{lsa_summary}\n")
    
    return results

# Streamlit app
def main():
    st.title("Text Analysis App")
    
    passage = st.text_area("Enter the passage for analysis:")
    
    if st.button("Analyze"):
        if passage:
            results = analyze_text(passage)
            for result in results:
                st.write(result)
        else:
            st.warning("Please provide the passage.")

if __name__ == "__main__":
    main()
