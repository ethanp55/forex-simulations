import time
import datetime
import pandas as pd


TICKER = 'ESU24.CME'
START_DATE = int(time.mktime(datetime.datetime(2019, 8, 4).timetuple()))
END_DATE = int(time.mktime(datetime.datetime(2024, 8, 4).timetuple()))
TIME_FRAME = '1d'


if __name__ == '__main__':
    request_url = f'https://query1.finance.yahoo.com/v7/finance/download/{TICKER}?period1={START_DATE}&period2={END_DATE}&interval={TIME_FRAME}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(request_url)
    df.to_csv(f'./data/{TICKER}_{TIME_FRAME}.csv')