# RNN Prediction of Apple Log Returns
1. Simple RNN
- Construct an Nth order lag filter with activation
2. Nth Order LSTM
- Same as above, but with long term memory
3. LSTM with Moving Averages
- N lookback + ~3 Moving Averages
4. LSTM with index comparison
- Lookback + 1-3 indexes (SPY, QQQ, XLK)
5. LSTM with Vol & RFR
- Lookback + GARCH, VIX, Risk Free Rate
6. LSTM with Events
- FED Meetings, Earning Calls, Jobs Reports, CPI, WWDC, Product launches
- No Sentiment, just boolean + number if applicable
7. LSTM with Events & Sentiment
- Finbert Sentiment Analysis
8. Kitchen Sink