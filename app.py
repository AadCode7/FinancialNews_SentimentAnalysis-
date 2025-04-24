import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

from data_collection import NewsCollector
from llm_components import LLMSentimentAnalyzer, SentimentAggregator

# Initialize components
news_collector = NewsCollector()
llm_analyzer = LLMSentimentAnalyzer()
sentiment_aggregator = SentimentAggregator()

def get_ftse100_data():
    """Get FTSE 100 index data for comparison"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    ftse = yf.download('^FTSE', start=start_date, end=end_date)
    return ftse['Close']

def analyze_articles(articles_df):
    """Analyze articles using LLM method"""
    results = []
    
    for _, article in articles_df.iterrows():
        text = news_collector.preprocess_article(article)
        
        # LLM analysis
        llm_result = llm_analyzer.analyze_text(text)
        print(f"LLM Result: {llm_result}")  # Debug print
        print(f"LLM Result keys: {llm_result.keys()}")  # Debug print
        
        # Convert sentiment_score to sentiment category
        sentiment_score = llm_result['sentiment_score']
        if sentiment_score > 0.3:
            sentiment = 'positive'
        elif sentiment_score < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        results.append({
            'date': article['publishedAt'],
            'title': article['title'],
            'sentiment': sentiment,
            'confidence': llm_result['confidence'],
            'sentiment_score': sentiment_score,
            'sectors': llm_result['sectors'],
            'key_phrases': llm_result['key_phrases']
        })
    
    return pd.DataFrame(results)

def plot_sentiment_timeseries(sentiment_df, ftse_data):
    """Create interactive plot of sentiment over time"""
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=sentiment_df['date'],
        y=sentiment_df['sentiment_score'],
        name='Sentiment Score',
        line=dict(color='blue'),
        mode='lines+markers',
        text=sentiment_df['title'],
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    # Add FTSE 100 line (normalized)
    ftse_normalized = (ftse_data - ftse_data.min()) / (ftse_data.max() - ftse_data.min())
    fig.add_trace(go.Scatter(
        x=ftse_data.index,
        y=ftse_normalized,
        name='FTSE 100 (Normalized)',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Sentiment Analysis vs FTSE 100',
        xaxis_title='Date',
        yaxis_title='Score',
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    return fig

def main():
    st.title('UK Economic Sentiment Analyzer')
    
    # Sidebar controls
    st.sidebar.header('Analysis Parameters')
    days_back = st.sidebar.slider('Number of days to analyze', 1, 30, 7)
    
    # Fetch and analyze articles
    with st.spinner('Fetching and analyzing articles...'):
        articles_df = news_collector.get_news_articles(days_back=days_back)
        if not articles_df.empty:
            results_df = analyze_articles(articles_df)
            ftse_data = get_ftse100_data()
            
            # Display results
            st.subheader('Recent Articles Analysis')
            st.dataframe(results_df[['date', 'title', 'sentiment', 'confidence', 'sectors']])
            
            # Plot sentiment timeseries
            st.subheader('Sentiment Analysis Over Time')
            fig = plot_sentiment_timeseries(results_df, ftse_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector analysis
            st.subheader('Sector-wise Sentiment')
            sector_sentiments = sentiment_aggregator.get_sector_sentiments(
                results_df.to_dict('records')
            )
            sector_df = pd.DataFrame(
                list(sector_sentiments.items()),
                columns=['Sector', 'Sentiment Score']
            )
            st.bar_chart(sector_df.set_index('Sector'))
            
            # Summary statistics
            st.subheader('Summary Statistics')
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    'Average Sentiment Score',
                    f"{results_df['sentiment_score'].mean():.2f}"
                )
            with col2:
                st.metric(
                    'Sentiment Volatility',
                    f"{results_df['sentiment_score'].std():.2f}"
                )
            
            # Key phrases analysis
            st.subheader('Key Phrases Analysis')
            all_phrases = [phrase for phrases in results_df['key_phrases'] for phrase in phrases]
            phrase_counts = pd.Series(all_phrases).value_counts().head(10)
            st.bar_chart(phrase_counts)
            
        else:
            st.error('No articles found for the selected period.')

if __name__ == '__main__':
    main() 