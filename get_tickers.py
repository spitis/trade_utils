import pandas as pd
import numpy as np

_tickers = None

def get_tickers ():
    global _tickers
    if _tickers is not None:
        return _tickers

    url_NSDQ = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
    url_NYSE = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"
    nsdq = pd.read_table(url_NSDQ,sep=",")
    nyse = pd.read_table(url_NYSE,sep=",")
    tickers = pd.concat([nsdq,nyse])

    def dollar_to_int (dollar_string):
        try:
            parsed = int(float(dollar_string[1:-1])*1000)
            if dollar_string[-1] == 'B':
                parsed *= 1000
            return parsed
        except:
            return np.NaN
    tickers = tickers.drop_duplicates("Name")
    tickers = tickers[["Symbol","MarketCap","Sector","industry"]]
    tickers.MarketCap = tickers.MarketCap.apply(dollar_to_int)
    tickers = tickers[np.isfinite(tickers.MarketCap)]

    _tickers = tickers.reset_index()[["Symbol","MarketCap","Sector","industry"]]
    return _tickers
