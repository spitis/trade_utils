import numpy as np
import pandas as pd
import datetime as dt

def convert_to_nextOpen(df):
    df['Next Open'] = df['Open'].shift(-1)
    df = df[np.isfinite(df['Next Open'])]
    df = df[['High','Low','Close','Next Open','Volume']]
    return df

def prepareX_sd4(date,sp500,min_dollar_vol, opens, yahoostore):
    keys = yahoostore['keys']
    sampledays = 641
    pastdays = 640

    tickers = []
    x = []
    x_raw = []

    today = None
    if opens is not None:
        today = True

    if today:
        startdate = sp500.index[-640]
        enddate = sp500.index[-1]
    else:
        try:
            loc = sp500.index.get_loc(date)
        except:
            return [], []

        startdate = sp500.index[loc - pastdays]
        enddate   = sp500.index[loc]

    spdf = sp500[startdate:enddate].copy()

    if today:
        new_row={'Open': opens[opens.Ticker.eq('^GSPC')]['Open'][0], 'High': 0, 'Low': 0, 'Close': 0, 'Volume':0}
        new_row = pd.DataFrame(data=new_row,index=[dt.date.today()])
        new_row = new_row[['Open','High','Low','Close','Volume']]
        spdf = spdf.combine_first(new_row)

    spdf = convert_to_nextOpen(spdf)
    spdf = spdf.as_matrix()
    x_sp = prepareX(np.reshape(spdf,(1,640,5)))

    lenkeys = len(keys)

    for idx, i in enumerate(keys):
        print("Preparing " + i + ". Progress: " + str(idx/lenkeys) +"%", end="\r")
        df = yahoostore['/stocks' + i]
        df = df[startdate:enddate].copy()
        try:
            if today and opens[opens.Ticker.eq(i[1:])]['Open'].values[0] > 0:
                if pastdays != len(df):
                    continue
                new_row={'Open': opens[opens.Ticker.eq(i[1:])]['Open'].values[0], 'High': 0, 'Low': 0, 'Close': 0, 'Volume':0}
                new_row = pd.DataFrame(data=new_row,index=[dt.date.today()])
                new_row = new_row[['Open','High','Low','Close','Volume']]
                df = df.combine_first(new_row)
            else:
                if sampledays != len(df):
                    continue
        except:
            if sampledays != len(df):
                continue

        df = convert_to_nextOpen(df)
        df = df.as_matrix()

        x_raw.append(df.copy())
        xs = prepareX(np.reshape(df,(1,640,5)))
        x.append(np.concatenate([xs,x_sp],axis=2))
        tickers.append(i)

    tickers, x = striplowvolume(min_dollar_vol, np.array(x_raw),np.array(tickers),np.array(x).reshape(len(x),8,8,5))
    return tickers, x

#takes an nparray Nx640x5, and returns xarray of size Nx8x4x5
def prepareX(arr):
    arr[:,:,4] = adjustVol(arr[:,:,4]) #adjusts volume based on mean volume
    out = np.zeros((arr.shape[0],8,4,5))
    out[:,0] = differX(arr[:,-5:])
    for i in range(1,8):
        arr = averX(arr)
        out[:,i] = differX(arr[:,-5:])
    return out

#takes an nparray Nx2Mx5 and returns NxMx5 array [open, high, low, close, adj Vol]
def averX(arr):
    out = np.zeros((arr.shape[0],arr.shape[1]//2,arr.shape[2]))
    for i in range(0,arr.shape[1],2):
        out[:,i//2,3] = arr[:,i+1,3] #NEXT OPEN gets the second NEXT OPEN
        out[:,i//2,0] = np.maximum(arr[:,i,0],arr[:,i+1,0]) #high gets the maximum
        out[:,i//2,1] = np.minimum(arr[:,i,1],arr[:,i+1,1]) #low gets the minimum
        out[:,i//2,2] = arr[:,i+1,2] #close gets the second close
        out[:,i//2,4] = (arr[:,i,4] + arr[:,i+1,4]) / 2 #volume gets averaged
    return out

#takes an nparray NxMx5 and returns NxM-1x5 with differences
def differX(arr):
    later9 = arr[:,1:]
    early9 = arr[:,:-1]
    return (arr[:,1:] - arr[:,:-1]) / arr[:,:-1]

def adjustVol(arr):
    avgs = np.nanmean(arr,axis=1)
    avgs[avgs == 0] = 1 #cure instances where all volume is 0
    arr = (arr.T / avgs).T #first normalize the array according to the average
    ones = np.ones(arr.shape)
    arr[np.where(np.isnan(arr))] = ones[np.where(np.isnan(arr))] #SHOULD NOT BE REQUIRED, and isn't necessary good because it will replace NaNs with avg volume, which may be a bad assumption. HOWEVER, it shouldn't happen, and at worst it will add some noise.
    arr += .5 # here we are adding .5 to avoid any really large values due to 0 volume days when it gets differenced
    return arr

def striplowvolume(mindollarvol, raw, tickers, x):
    means = np.mean(raw, axis=1)
    avgopens = means[:,3]
    avgvols = means[:,4]
    dollarvols = avgopens * avgvols
    return tickers[dollarvols > mindollarvol], x[dollarvols > mindollarvol]
