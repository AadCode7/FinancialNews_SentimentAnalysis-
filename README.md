# LLM-Powered Sentiment Analyzer for Financial News

This project implements a sentiment analysis tool that analyzes financial news articles to generate sentiment scores related to recession fears in the UK economy.

## Project Overview

The tool uses a combination of approaches:
1. LLM-based sentiment analysis
2. Naive Bayes classifier
3. Dictionary-based approach
4. Rule-based systems

## Features

- Real-time financial news collection from NewsAPI
- Multiple sentiment analysis approaches
- Time series visualization of sentiment trends
- Sector-specific sentiment analysis
- Comparison with market indices

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API keys:
```
NEWS_API_KEY=your_news_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure

- `app.py`: Main Streamlit application
- `llm_components.py`: LLM-related functions and utilities
- `sentiment_analysis.py`: Sentiment analysis implementations
- `data_collection.py`: News API integration and data collection
- `utils.py`: Utility functions and helper methods

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The application will:
   - Fetch recent financial news articles
   - Analyze sentiment using multiple approaches
   - Generate time series visualizations
   - Provide sector-specific insights

## Sentiment Analysis Approaches

1. **LLM-based Analysis**: Uses OpenAI's GPT models for nuanced sentiment understanding
2. **Naive Bayes**: Traditional machine learning approach
3. **Dictionary-based**: Uses financial sentiment lexicons
4. **Rule-based**: Implements negation and diminisher rules

## Output

- Daily sentiment scores
- Time series visualization
- Sector-specific sentiment trends
- Comparison with market indices

## Contributing

Feel free to submit issues and enhancement requests. 