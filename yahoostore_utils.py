"""
class yahoostore:
    stores a pandas dataframe that holds all yahoo stock dataframe

    build() - builds initial dataframe. WARNING: will overwrite old dataframe
    soft_update() - updates the dataframe with current data
    prepareX_date - prepares X sample for specific date
"""

import numpy as np
import datetime as dt
import pandas as pd
from pandas_datareader.data import DataReader
from trade_utils import get_tickers
from trade_utils.data_prep import prepare_sd4
from trade_utils.final_models import sd4

market_close = dt.time(16,45) #NOTE: Offset for yahoo's time delay
market_open  = dt.time(9,30)

def adjust(df):
    """
    adjusts yahoo data for stock splits and deletes adjusted close column
    """

    df = df[np.isfinite(df['Open'])]
    df = df[np.isfinite(df['High'])]
    df = df[np.isfinite(df['Low'])]
    df = df[np.isfinite(df['Close'])]
    df = df[np.isfinite(df['Adj Close'])]
    df = df[np.isfinite(df['Volume'])]

    ssplits = df['Adj Close'] / df['Close']
    for i in ['Open','High','Low','Close']:
        df[i] = df[i] * ssplits
    df.pop('Adj Close')
    df = df[np.isfinite(df['Open'])]
    df = df[np.isfinite(df['High'])]
    df = df[np.isfinite(df['Low'])]
    df = df[np.isfinite(df['Close'])]
    df = df[np.isfinite(df['Volume'])]
    return df

def strip_keys(keys):
    """#gets rid of the leading '/' in the keys list"""
    return [x[1:] for x in keys]

class yahoostore:
    def __init__(self, target):
        self.yahoostore = pd.HDFStore(target)
        self.sp500 = adjust(DataReader('^GSPC',  'yahoo', dt.datetime(1950,1,1), dt.date.today() + dt.timedelta(days=1)))
        self.opens = None

    def close(self):
        self.yahoostore.close()

    def build(self):
        """
        builds (or rebuilds) initial yahoostore at target
        """
        tickers = get_tickers.get_tickers()
        yahoostore = self.yahoostore
        keys = []
        for i in tickers.index:
            try:
                df = DataReader(tickers.ix[i].Symbol,  'yahoo', dt.datetime(1950,1,1), dt.date.today() + dt.timedelta(days=1))
            except:
                continue

            if len(df) > 0: #at least 1 day of data
                yahoostore['/stocks/'+tickers.ix[i].Symbol] = adjust(df)
                yahoostore.get_storer('/stocks/'+tickers.ix[i].Symbol).attrs.MarketCap = tickers.ix[i].MarketCap
                yahoostore.get_storer('/stocks/'+tickers.ix[i].Symbol).attrs.Sector = tickers.ix[i].Sector
                yahoostore.get_storer('/stocks/'+tickers.ix[i].Symbol).attrs.Industry = tickers.ix[i].industry
                keys.append('/'+tickers.ix[i].Symbol)
                print(tickers.ix[i].Symbol + " OK!", end=" ")
            else:
                print(tickers.ix[i].Symbol + " BLEH!", end=" ")
        yahoostore['keys'] = pd.Series(keys)
        yahoostore['last_update'] = pd.Series(dt.datetime.today())

    def last_updated(self):
        return self.yahoostore['last_update'][0]

    def up_to_date(self):
        if self.yahoo_updated_today() and self.last_updated() >= self.sp500.index[-1]:
            return True
        else:
            return False

    def yahoo_updated_today(self):
        self.sp500 = adjust(DataReader('^GSPC',  'yahoo', dt.datetime(1950,1,1), dt.date.today() + dt.timedelta(days=1)))
        if dt.datetime.today().day == self.sp500.index[-1].day:
            return True
        else:
            return False

    def is_sp500_business_day(self, date):
        try:
            day = self.sp500.ix[date]
            return True
        except:
            return False

    def is_today(self, date):
        if (date - dt.date.today()).days == 0:
            return True
        else:
            return False

    def is_market_open(self):
        now = dt.datetime.today().time()
        if now > market_open and now < market_close:
            url = "http://finance.yahoo.com/d/quotes.csv?s=^GSPC&f=sd1"
            data = pd.read_table(url,sep=",",header=None,names=['Ticker','Last Trade'],index_col=False)
            last_trade = dt.datetime.strptime(data['Last Trade'][0],'%m/%d/%Y')

            if self.is_today(last_trade.date()):
                return True
            else:
                return False
        else:
            return False

    def soft_update(self):
        """updates each ticker in yahoostore
        does not add new tickers or update attributes (e.g., industry)
        """

        last_day_sp500 = self.sp500.index[-1]
        for key in self.yahoostore['keys']:
            last_day_stock = self.yahoostore['/stocks' + key].index[-1]
            if (last_day_sp500 - last_day_stock).days == 0:
                None
            else:
                print('Updating ' + key, end=" ")
                self._soft_update_ticker(key, last_day_stock - dt.timedelta(days=1))
                print('Done!', end=" ")

        self.yahoostore['last_update'] = pd.Series(dt.datetime.today())

        return

    def _soft_update_ticker(self, key, startdate):
        try:
            stock_update = DataReader(key[1:],  'yahoo', startdate, dt.date.today() + dt.timedelta(days=1))
        except:
            return
        self.yahoostore['/stocks' + key] = self.yahoostore['/stocks' + key].combine_first(stock_update)
        return

    def prepareX_date(self, date=dt.date.today(), data_format="sd4", min_dollar_vol=250000, opens=None):
        """
        Prepares X (input matrix) for a specific day in a given data_format

        Returns (tickers, X), where tickers and X are np-arrays
        """
        if self.is_today(date):
            if self.is_market_open() and not self.yahoo_updated_today():
                opens = self.get_opens()

        if not self.up_to_date():
            self.soft_update()

        if opens is None and not self.is_sp500_business_day(date):
            return [], []

        if data_format == "sd4":
            return prepare_sd4.prepareX_sd4(date, self.sp500, min_dollar_vol, opens, self.yahoostore)
        else:
            print("Bad data format!")
            return

    def prepareY_date(self, date, tickers, data_format="one_day_open"):
        """
        Prepares y (output matrix) for a specific day given a list of tickers and a given data_format
        """
        if data_format == "one_day_open":
            res = []
            for i in range(len(tickers)):
                stock_df = self.yahoostore['/stocks' + tickers[i]]
                loc = stock_df.index.get_loc(date)
                try:
                    delta = (stock_df.ix[loc+1]['Open'] - stock_df.ix[loc]['Open']) / stock_df.ix[loc]['Open']
                except:
                    delta = np.NaN
                res.append(delta)
            return np.array(res)
        if data_format == "sd4":
            return []
        else:
            print("Bad data format!")
            return

    def prepareXY_range(self, timestamp_range, formatX="sd4", formatY="one_day_open"):
        """
        Returns dictionary of {date: (tickers, X, y)}
        Should not be used for very long periods at once as dictionary will blow-up in size
        ----
        Should not be used for building the big sample, since its not vectorized / in a for loop

        """
        xy = {}
        for i in timestamp_range:
            print("Working on " + str(i), end="\r")
            tickers, X = self.prepareX_date(i.date(),formatX)
            if len(tickers) > 0:
                y = self.prepareY_date(i.date(),tickers,formatY)
                tickers = tickers[np.isfinite(y)]
                X = X[np.isfinite(y)]
                y = y[np.isfinite(y)]

                xy[str(i.date())] = (tickers, X, y)
        return xy

    def predict_today_sd4(self, date=dt.date.today()):
        tickers, X = self.prepareX_date(date)
        if len(tickers) is 0:
            print("Invalid date")
            return
        return sd4.top_tickers_with_spreads(tickers, X)

    # Gets the current day's Open, along with spread information and Market Cap
    def get_opens(self):
        df = get_opens(self.yahoostore['keys'])
        self.opens = df
        return df

def get_opens(keys):
    url = "http://finance.yahoo.com/d/quotes.csv?s=^GSPC&f=sobaj1"
    df = pd.read_table(url,sep=",",header=None,names=['Ticker','Open','Bid','Ask','MarketCap'],index_col=False)
    for i in range(6):
        ticker_query = "".join(list(keys)[i*1000:min((i+1)*1000,len(keys))])[1:].replace('/','+')

        #http://www.jarloo.com/yahoo_finance/
        #o = open
        #a = ask
        #b = bid
        #j1 = market cap
        url = "http://finance.yahoo.com/d/quotes.csv?s=" + ticker_query + "&f=sobaj1"
        data = pd.read_table(url,sep=",",header=None,names=['Ticker','Open','Bid','Ask','MarketCap'],index_col=False)
        df = pd.concat([df,data])

    df = df.reset_index()[['Ticker','Open','Bid','Ask','MarketCap']]
    return df
