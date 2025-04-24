import os
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class NewsCollector:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.newsapi = NewsApiClient(api_key=self.api_key)
        
    def get_news_articles(self, days_back=7, max_articles=100):
        """
        Fetch news articles about UK economy and financial markets
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_articles = []
        
        # Keywords related to UK economy and recession
        keywords = [
            'UK economy', 'UK recession', 'Bank of England', 'UK inflation',
            'UK interest rates', 'UK GDP', 'UK financial markets',
            'UK economic growth', 'UK economic outlook'
        ]
        
        for keyword in keywords:
            try:
                articles = self.newsapi.get_everything(
                    q=keyword,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=min(100, max_articles)
                )
                
                if articles['status'] == 'ok':
                    all_articles.extend(articles['articles'])
            except Exception as e:
                print(f"Error fetching articles for keyword '{keyword}': {str(e)}")
        
        # Convert to DataFrame and clean
        df = pd.DataFrame(all_articles)
        if not df.empty:
            df = df.drop_duplicates(subset=['title'])
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            df = df.sort_values('publishedAt', ascending=False)
        
        return df
    
    def get_daily_articles(self, date):
        """
        Get articles for a specific date
        """
        start_date = datetime.combine(date, datetime.min.time())
        end_date = start_date + timedelta(days=1)
        
        articles = self.newsapi.get_everything(
            q='UK economy OR UK recession OR Bank of England',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy'
        )
        
        if articles['status'] == 'ok':
            df = pd.DataFrame(articles['articles'])
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            return df
        return pd.DataFrame()
    
    def get_historical_articles(self, start_date, end_date):
        """
        Get articles for a date range
        """
        articles = self.newsapi.get_everything(
            q='UK economy OR UK recession OR Bank of England',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy'
        )
        
        if articles['status'] == 'ok':
            df = pd.DataFrame(articles['articles'])
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            return df
        return pd.DataFrame()
    
    def preprocess_article(self, article):
        """
        Combine title and description for analysis
        """
        text = f"{article['title']}. {article['description']}"
        if 'content' in article and article['content']:
            text += f" {article['content']}"
        return text 