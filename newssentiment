import yfinance as yf
import finnhub
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import numpy as np
import pandas as pd
import backtrader as bt  
import datetime
import nltk
import requests

# API keys
FINNHUB_API_KEY = "..."  # Replace with your API key
THENEWSAPI_KEY = "..."  # Replace with your API key
TICKER = "ANY STOCK"
FROM_DATE = "2024-11-01"
TO_DATE = "2024-12-30"

# Fetch historical stock data
def get_stock_data(ticker, from_date, to_date):
    data = yf.download(ticker, start=from_date, end=to_date)
    data.columns = [col if isinstance(col, str) else col[0] for col in data.columns]
    return data

# Fetch news sentiment using Finnhub
def get_finnhub_sentiment(ticker):
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    try:
        news = finnhub_client.company_news(symbol=ticker, _from=FROM_DATE, to=TO_DATE)
    except Exception as e:
        print(f"Error fetching Finnhub news: {e}")
        return 0

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in news:
        try:
            sentiment = analyzer.polarity_scores(article['headline'])['compound']
            sentiments.append(sentiment)
        except:
            pass
    return np.mean(sentiments) if sentiments else 0

# Fetch news sentiment using TheNewsApi
def get_thenewsapi_sentiment(ticker):
    url = f"https://api.thenewsapi.com/v1/news/all?api_token={THENEWSAPI_KEY}&symbols={ticker}&published_after={FROM_DATE}&published_before={TO_DATE}&language=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news = response.json().get('data', [])
    except Exception as e:
        print(f"Error fetching TheNewsApi news: {e}")
        return 0

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in news:
        try:
            sentiment = analyzer.polarity_scores(article['title'])['compound']
            sentiments.append(sentiment)
        except:
            pass
    return np.mean(sentiments) if sentiments else 0

# Combined sentiment score from both sources
def get_combined_sentiment(ticker):
    finnhub_sentiment = get_finnhub_sentiment(ticker)
    thenewsapi_sentiment = get_thenewsapi_sentiment(ticker)
    
    combined_sentiment = np.mean([finnhub_sentiment, thenewsapi_sentiment])
    print(f"Finnhub Sentiment: {finnhub_sentiment}, TheNewsApi Sentiment: {thenewsapi_sentiment}, Combined Sentiment: {combined_sentiment}")
    return combined_sentiment

# Strategy for backtrader
class SentimentAndTechnicalStrategy(bt.Strategy):
    params = (('rsi_period', 14), ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9))

    def __init__(self):
        # Indicators
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close)
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)

        # Track buy and sell signals
        self.sentiment = get_combined_sentiment(TICKER)

    def next(self):
        self.sentiment = get_combined_sentiment(TICKER)

        if not self.position:  # Not in the market
            if self.sentiment > 0.2 and self.rsi < 50 and self.macd.macd[0] > self.macd.signal[0] and self.data.close[0] > self.sma[0]:
                self.buy()  # Create a buy order
                self.buy_price = self.data.close[0]  # Record entry price
                self.log(f"BUY: {self.data.close[0]}")

        elif self.position:
            if self.sentiment < -0.2 or self.data.close[0] <= self.buy_price * 0.98 or self.data.close[0] >= self.buy_price * 1.05:
                self.sell()  # Close position
                self.log(f"SELL: {self.data.close[0]}")

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

# Set up Cerebro
cerebro = bt.Cerebro()
cerebro.addstrategy(SentimentAndTechnicalStrategy)

data = bt.feeds.PandasData(dataname=get_stock_data(TICKER, FROM_DATE, TO_DATE))
cerebro.adddata(data)

cerebro.broker.setcash(100000.0)  
cerebro.addsizer(bt.sizers.PercentSizer, percents=20)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Plot the graph
cerebro.plot()
