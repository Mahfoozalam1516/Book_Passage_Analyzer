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

# Ensure NLTK data is available
nltk.data.path.append('./nltk_data')  # Adjust the path to your nltk_data folder

# Linking to Google Books API
def search_books(passage, api_key):
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
def analyze_text(passage, api_key):
    """Main function to analyze the text and display results in Streamlit"""
    st.title("Text Analysis Results")
    
    # 1. Word Count
    total_words = count_words(passage)
    total_words_no_stopwords = count_words_without_stopwords(passage)

    st.write(f"Total number of words: {total_words}")
    st.write(f"Total number of words (without stopwords): {total_words_no_stopwords}")
    
    # 2. Emotional Analysis
    emotion = analyze_emotion(passage)
    st.write(f"Predominant emotion: {emotion}\n")

    # 3. Book Search
    st.write("Possible books the passage might be from:")
    books = search_books(passage, api_key)
    if books:
        for item in books.get('items', [])[:3]:  # To get the first 3 possible books
            title = item['volumeInfo'].get('title', 'No title found')
            authors = item['volumeInfo'].get('authors', ['No authors found'])
            st.write(f"- Title: {title}")
            st.write(f"  Authors: {', '.join(authors)}\n")
    
    # 4. Summary using LSA
    lsa_summary = summarize_with_lsa(passage)
    st.write(f"Summary:\n{lsa_summary}\n")

# Streamlit UI
if __name__ == "__main__":
    st.sidebar.header("Input Passage")
    passage = st.sidebar.text_area("Enter your passage here:", height=200)
    api_key = st.sidebar.text_input("Enter your Google Books API key:")
    
    if st.sidebar.button("Analyze"):
        if passage and api_key:
            analyze_text(passage, api_key)
        else:
            st.warning("Please enter both a passage and an API key.")
