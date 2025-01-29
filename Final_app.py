import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import requests
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import toml

# Download required NLTK data 
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Load API key from .streamlit/config.toml
config = toml.load(".streamlit/config.toml")
api_key = config['general']['api_key']

def search_books(passage):
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
        st.error("Error: " + str(response.status_code))
        return None

def count_words(passage):
    words = word_tokenize(passage)
    return len([word for word in words if word not in string.punctuation])

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
    elif scores['neg'] > 0.1:  # Disgust threshold
        return 'disgust'
    elif scores['pos'] < 0.1:  # Fear threshold
        return 'fear'
    else:
        return 'neutral'

def summarize_with_lsa(passage, num_sentences):
    parser = PlaintextParser.from_string(passage, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)  # Summarize to specified number of sentences
    return ' '.join(str(sentence) for sentence in summary)

def analyze_text(passage, num_sentences):
    """Main function to analyze the text and print results"""
    st.write("=== Text Analysis Results ===\n")
    
    # 1. Word Count
    total_words = count_words(passage)
    st.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h4 style="margin: 0;">Total number of words:</h4>
        <p style="margin: 0;">{total_words}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Emotional Analysis
    emotion = analyze_emotion(passage)
    st.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h4 style="margin: 0;">Predominant emotion:</h4>
        <p style="margin: 0;">{emotion}</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Book Search
    st.markdown("<h4>Possible books the passage might be from:</h4>", unsafe_allow_html=True)
    books = search_books(passage)
    if books:
        for item in books.get('items', [])[:3]:
            title = item['volumeInfo'].get('title', 'No title found')
            authors = item['volumeInfo'].get('authors', ['No authors found'])
            st.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h4 style="margin: 0;">{title}</h4>
        <p style="margin: 0; color: gray;">Authors: {', '.join(authors)}</p>
    </div>
    """, unsafe_allow_html=True)

    # 4. Summary using LSA
    lsa_summary = summarize_with_lsa(passage, num_sentences)
    st.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h4 style="margin: 0;">Summary:</h4>
        <p style="margin: 0;">{lsa_summary}</p>
    </div>
    """, unsafe_allow_html=True)


# Streamlit UI
st.title("Text Analysis App")

passage = st.text_area("Enter the text you want to analyze:")

# Option to select number of summary sentences
num_sentences = st.slider("Select number of summary sentences:", min_value=1, max_value=5, value=2)

if st.button("Analyze"):
    if passage:
        analyze_text(passage, num_sentences)
    else:
        st.error("Please provide text.")
