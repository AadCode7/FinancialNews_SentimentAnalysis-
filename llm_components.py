import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

load_dotenv()

class LLMSentimentAnalyzer:
    def __init__(self):
        # Initialize OpenAI client with API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        self.system_prompt = """You are a financial analyst specializing in sentiment analysis of economic news. 
        Analyze the given text and determine the sentiment regarding recession fears in the UK economy.
        
        Consider the following aspects:
        1. Explicit mentions of recession or economic downturn
        2. Economic indicators and their trends (GDP, inflation, employment)
        3. Market sentiment and investor confidence
        4. Government policies and their impact
        5. Global economic factors affecting the UK
        6. Sector-specific impacts and concerns
        
        Return a JSON object with:
        - sentiment_score: float between -1.0 and 1.0 where:
          * -1.0 = extremely positive (no recession fears)
          * 0.0 = neutral/mixed
          * 1.0 = extremely negative (strong recession fears)
        - confidence: float between 0 and 1 (higher means more confident in the sentiment)
        - key_phrases: list of specific phrases from the text that support your sentiment analysis
        - sectors: list of economic sectors mentioned in the article
        - date: the date of the article in YYYY-MM-DD format
        
        Example response:
        {
            "sentiment_score": 0.75,
            "confidence": 0.85,
            "key_phrases": ["rising inflation", "economic slowdown", "recession warning", "market uncertainty"],
            "sectors": ["banking", "retail", "manufacturing"],
            "date": "2024-03-15"
        }
        """
    
    def analyze_text(self, text: str, date: str = None) -> Dict[str, Any]:
        try:
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
                
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Date: {date}\n\nText: {text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"LLM Analysis Result: {result}")  # Debug print
            return result
        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.5,
                "key_phrases": [],
                "sectors": [],
                "date": date
            }
    
    def batch_analyze(self, texts: List[str], dates: List[str] = None) -> List[Dict[str, Any]]:
        results = []
        for i, text in enumerate(texts):
            date = dates[i] if dates and i < len(dates) else None
            result = self.analyze_text(text, date)
            results.append(result)
        return results

class SentimentAggregator:
    def __init__(self):
        self.sentiment_history = []
        self.sentiment_mapping = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
    
    def aggregate_daily_sentiment(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sentiment_results:
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "article_count": 0,
                "key_phrases": [],
                "sectors": set()
            }
        
        # Get the date from the first result
        date = sentiment_results[0]["date"]
        
        # Calculate weighted average sentiment
        weighted_sum = 0
        total_confidence = 0
        all_key_phrases = []
        all_sectors = set()
        
        for result in sentiment_results:
            sentiment_score = result["sentiment_score"]
            confidence = result["confidence"]
            weighted_sum += sentiment_score * confidence
            total_confidence += confidence
            all_key_phrases.extend(result["key_phrases"])
            all_sectors.update(result["sectors"])
            
            print(f"Processing result - Score: {sentiment_score}, Confidence: {confidence}")  # Debug print
        
        if total_confidence == 0:
            avg_sentiment = 0.0
        else:
            avg_sentiment = weighted_sum / total_confidence
        
        print(f"Daily aggregated score for {date}: {avg_sentiment}")  # Debug print
        
        return {
            "date": date,
            "sentiment_score": avg_sentiment,
            "confidence": total_confidence / len(sentiment_results),
            "article_count": len(sentiment_results),
            "key_phrases": list(set(all_key_phrases)),  # Remove duplicates
            "sectors": list(all_sectors)
        }
    
    def get_sector_sentiments(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, float]:
        sector_sentiments = {}
        sector_counts = {}
        
        for result in sentiment_results:
            for sector in result["sectors"]:
                if sector not in sector_sentiments:
                    sector_sentiments[sector] = 0.0
                    sector_counts[sector] = 0
                
                sentiment_score = result["sentiment_score"]
                confidence = result["confidence"]
                sector_sentiments[sector] += sentiment_score * confidence
                sector_counts[sector] += 1
        
        # Calculate weighted average sentiment per sector
        sector_averages = {}
        for sector, total in sector_sentiments.items():
            count = sector_counts[sector]
            if count > 0:
                sector_averages[sector] = total / count
                print(f"Sector {sector} average sentiment: {sector_averages[sector]}")  # Debug print
        
        return sector_averages 