A Stock Predicting algorithm using News sentiment and overall stock data. Built in Python, The Algorithm works by analyzing news and stock price data over a desired timeframe, then it makes predictions on optimal buy and sell indicators as well as a prediction for the next day. 


for the Sentiment.py 
1. Get api Keys for NewsAPI and Finnhub(you can replace this with your own API or any other API, I used these because they are free).
2. Analyze any stock you want. Works with Index 
3. Select a timeframe to train the model from(a shorter timeframe leads to faster training and vice versa)
4. See accurate buy and sell calls on the stock
5. and always make sure to PIP install all required libraries
-When the MACD Value crosses the signal value from the top to bottom, it indicates a sell, when the MACD value crosses the signal value from the bottom to top that indicates a buy
-Simple Moving Average(SMA) is an average of the trend of the stock 
-The gray and red rectangles indicate the volume that the stock was sold at. 

for simplepredict.py
1. Analyze any stock you wan't
2. Select a timeframe to train the model from(a shorter timeframe leads to faster training and vice versa)
3. Increasing prediction_days leads to longer training times, but will also make long term trends more accurate, short term trends less accurate.
4. Increasing Epochs will also lead to longer training times, but will also make long term and short term trends more accurate
-The model prompts you with an initial investment, and then it compares a buy and hold with the algorithms prediciton vs the actual market and somtimes it outperforms market. For example with MSFT and SPY it outperforms.
-it also prompts how many days you want to predict into the future, 5 days is the sweet spot and most accurate, anything past that leads to innacurate results.
-shows % accuracy of the model, the longer it trains usually the better accuracy. MSFT reaches 97%+ even with short training. 


<img width="1280" alt="image" src="https://github.com/user-attachments/assets/29d9cc1e-3de0-4fb7-b7fc-d43563a23b63" />

<img width="1193" alt="image" src="https://github.com/user-attachments/assets/66bee0a5-20e7-4eb8-81fb-4e99995c0299" />

