import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from llm_components import LLMSentimentAnalyzer, SentimentAggregator

class TimeSeriesAnalyzer:
    def __init__(self, index_symbol: str = "^FTSE"):
        self.sentiment_analyzer = LLMSentimentAnalyzer()
        self.sentiment_aggregator = SentimentAggregator()
        self.index_symbol = index_symbol
        
    def get_index_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch index returns for the specified period"""
        index = yf.Ticker(self.index_symbol)
        data = index.history(start=start_date, end=end_date)
        returns = data['Close'].pct_change()
        return returns
    
    def analyze_time_period(self, articles: List[Dict[str, Any]], 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """Analyze sentiment for articles over a time period and create time series"""
        # Group articles by date
        articles_by_date = {}
        for article in articles:
            date = article.get('date', datetime.now().strftime("%Y-%m-%d"))
            if date not in articles_by_date:
                articles_by_date[date] = []
            articles_by_date[date].append(article)
        
        # Analyze sentiment for each day
        daily_sentiments = []
        for date, day_articles in articles_by_date.items():
            # Extract text from articles
            texts = [article['text'] for article in day_articles]
            dates = [date] * len(texts)
            
            # Analyze sentiment
            sentiment_results = self.sentiment_analyzer.batch_analyze(texts, dates)
            print(f"Sentiment results for {date}: {sentiment_results}")  # Debug print
            
            daily_sentiment = self.sentiment_aggregator.aggregate_daily_sentiment(sentiment_results)
            print(f"Daily aggregated sentiment: {daily_sentiment}")  # Debug print
            daily_sentiments.append(daily_sentiment)
        
        # Create DataFrame
        sentiment_df = pd.DataFrame(daily_sentiments)
        print(f"Sentiment DataFrame columns: {sentiment_df.columns}")  # Debug print
        
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        
        # Get index returns
        returns = self.get_index_returns(start_date, end_date)
        
        # Combine sentiment and returns
        combined_df = pd.DataFrame({
            'sentiment_score': sentiment_df['sentiment_score'],
            'index_returns': returns
        })
        
        return combined_df
    
    def plot_sentiment_vs_returns(self, data: pd.DataFrame, 
                                title: str = "Sentiment vs Index Returns"):
        """Plot sentiment scores against index returns"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment Score', color=color)
        ax1.plot(data.index, data['sentiment_score'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot returns
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Index Returns', color=color)
        ax2.plot(data.index, data['index_returns'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(title)
        fig.tight_layout()
        plt.show()
    
    def calculate_correlation(self, data: pd.DataFrame, 
                           lag_days: int = 0) -> float:
        """Calculate correlation between sentiment and returns with optional lag"""
        if lag_days > 0:
            sentiment_shifted = data['sentiment_score'].shift(lag_days)
            returns = data['index_returns']
        else:
            sentiment_shifted = data['sentiment_score']
            returns = data['index_returns']
        
        # Remove NaN values
        valid_data = pd.concat([sentiment_shifted, returns], axis=1).dropna()
        
        return valid_data['sentiment_score'].corr(valid_data['index_returns'])

def main():
    # Example usage
    analyzer = TimeSeriesAnalyzer()
    
    # Example articles (replace with actual article data)
    articles = [
        {
            'text': 'UK economy shows signs of recovery as inflation eases...',
            'date': '2024-03-15'
        },
        {
            'text': 'Recession fears grow as manufacturing output declines...',
            'date': '2024-03-15'
        },
        # Add more articles...
    ]
    
    try:
        # Analyze time period
        start_date = '2024-01-01'
        end_date = '2024-03-15'
        data = analyzer.analyze_time_period(articles, start_date, end_date)
        
        # Plot results
        analyzer.plot_sentiment_vs_returns(data)
        
        # Calculate correlations with different lags
        for lag in [0, 1, 2, 3, 4, 5]:
            correlation = analyzer.calculate_correlation(data, lag)
            print(f"Correlation with {lag}-day lag: {correlation:.3f}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 