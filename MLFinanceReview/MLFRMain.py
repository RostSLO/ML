'''
Created on January 22, 2021

@author rboruk
'''

import yfinance as yf
from SearchPeak import SearchPeak

import pandas as pd

tickerName = "AMZN"

msft = yf.Ticker(tickerName)
print(msft)
"""
returns
<yfinance.Ticker object at 0x1a1715e898>
"""

# get stock info
#msft.info

"""
returns:
{
 'quoteType': 'EQUITY',
 'quoteSourceName': 'Nasdaq Real Time Price',
 'currency': 'USD',
 'shortName': 'Microsoft Corporation',
 'exchangeTimezoneName': 'America/New_York',
  ...
 'symbol': 'MSFT'
}
"""

# get historical market data, here max is 5 years.
data = msft.history(period="3y")

data.to_csv(tickerName + '.csv')

transform = SearchPeak(tickerName)

if transform: print("All good")
else: print("something doesn't work")