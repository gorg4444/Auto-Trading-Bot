import os
import requests
import logging
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

class NewsProcessor:
    """
    Analyzes news headlines and fetches rich historical news from Polygon.io.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.api_token = os.getenv("POLYGON_API_KEY")

    def analyze_sentiment(self, text):
        """Analyzes text and returns a simple sentiment score."""
        if not text: # Handle cases where text might be None or empty
            return 'neutral'
        score = self.analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def get_historical_news(self, symbols, start_date, end_date):
        """
        Fetches historical news for a list of symbols from Polygon.io,
        handling rate limits by fetching data in monthly chunks.
        """
        if not self.api_token:
            logging.warning("Polygon.io API token not found. Cannot fetch historical news.")
            return []

        all_news = []
        seen_article_ids = set()

        current_date = start_date
        while current_date < end_date:
            next_month = current_date + timedelta(days=30)
            range_end = min(next_month, end_date)
            
            start_str = current_date.strftime('%Y-%m-%d')
            end_str = range_end.strftime('%Y-%m-%d')

            logging.info(f"Fetching news for {len(symbols)} symbols from {start_str} to {end_str}...")
            
            # --- THIS IS THE FIX ---
            # The URL parameter for multiple tickers is 'ticker.any_of', not 'ticker'.
            # The requests library handles comma encoding automatically.
            url = (
                f"https://api.polygon.io/v2/reference/news"
                f"?published_utc.gte={start_str}"
                f"&published_utc.lte={end_str}"
                f"&ticker.any_of={','.join(symbols)}"
                f"&limit=1000"
                f"&apiKey={self.api_token}"
            )

            try:
                # Respect the 5 calls/minute rate limit of the free tier
                time.sleep(12) 

                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for article in data.get("results", []):
                    if article.get('id') not in seen_article_ids:
                        text_to_analyze = article.get('title', '') + " " + article.get('description', '')
                        sentiment = self.analyze_sentiment(text_to_analyze)
                        
                        processed_article = {
                            'created_at': article['published_utc'],
                            'headline': article.get('title', 'No Title'),
                            'snippet': article.get('description', ''),
                            'symbols': article.get('tickers', []),
                            'sentiment': sentiment
                        }
                        all_news.append(processed_article)
                        seen_article_ids.add(article.get('id'))

            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to fetch news from Polygon.io for range {start_str} - {end_str}: {e}")
            
            current_date = next_month

        logging.info(f"Total historical news articles fetched: {len(all_news)}")
        return all_news

