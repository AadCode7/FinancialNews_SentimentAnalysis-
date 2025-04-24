import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import spacy
import pandas as pd
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class TraditionalSentimentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        self.financial_lexicon = self._load_financial_lexicon()
        self.negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere'}
        self.diminisher_words = {'slightly', 'somewhat', 'partially', 'marginally', 'moderately'}
        
    def _load_financial_lexicon(self):
        # Basic financial sentiment lexicon
        return {
            'positive': {'growth', 'recovery', 'expansion', 'bullish', 'optimistic', 'strong', 'improve'},
            'negative': {'recession', 'decline', 'contraction', 'bearish', 'pessimistic', 'weak', 'deteriorate'},
            'intensifiers': {'significantly', 'dramatically', 'substantially', 'severely', 'extremely'},
            'diminishers': {'slightly', 'somewhat', 'partially', 'marginally', 'moderately'}
        }
    
    def preprocess_text(self, text):
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)
    
    def naive_bayes_analysis(self, texts, labels=None):
        if labels is None:
            # Use TextBlob for initial sentiment if no labels provided
            labels = [1 if TextBlob(text).sentiment.polarity > 0 else 0 for text in texts]
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        clf = MultinomialNB()
        clf.fit(X, labels)
        
        return clf, vectorizer
    
    def dictionary_based_analysis(self, text):
        tokens = word_tokenize(text.lower())
        sentiment_score = 0
        negation = False
        intensifier = 1.0
        
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                negation = True
            elif token in self.diminisher_words:
                intensifier = 0.5
            elif token in self.financial_lexicon['positive']:
                sentiment_score += 1 * intensifier * (-1 if negation else 1)
            elif token in self.financial_lexicon['negative']:
                sentiment_score -= 1 * intensifier * (-1 if negation else 1)
            elif token in self.financial_lexicon['intensifiers']:
                intensifier = 2.0
            else:
                negation = False
                intensifier = 1.0
        
        return sentiment_score
    
    def rule_based_analysis(self, text):
        doc = self.nlp(text)
        sentiment_score = 0
        
        for sent in doc.sents:
            # Check for negation patterns
            for token in sent:
                if token.dep_ == 'neg':
                    sentiment_score -= 0.5
                
                # Check for economic indicators
                if token.text.lower() in {'gdp', 'inflation', 'unemployment', 'interest rates'}:
                    sentiment_score += 0.3 if token.sentiment > 0 else -0.3
        
        return sentiment_score
    
    def extract_sectors(self, text):
        doc = self.nlp(text)
        sectors = set()
        
        # Common financial sectors
        financial_sectors = {
            'technology', 'finance', 'healthcare', 'energy', 'consumer', 'industrial',
            'materials', 'utilities', 'real estate', 'telecommunications'
        }
        
        for token in doc:
            if token.text.lower() in financial_sectors:
                sectors.add(token.text.lower())
        
        return list(sectors)
    
    def analyze_article(self, text):
        # Combine multiple approaches
        dictionary_score = self.dictionary_based_analysis(text)
        rule_score = self.rule_based_analysis(text)
        sectors = self.extract_sectors(text)
        
        # Weighted average of scores
        final_score = (dictionary_score * 0.6 + rule_score * 0.4)
        
        # Convert to sentiment
        if final_score > 0.2:
            sentiment = "positive"
        elif final_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "confidence": abs(final_score),
            "sectors": sectors,
            "raw_scores": {
                "dictionary": dictionary_score,
                "rule_based": rule_score
            }
        } 