import pandas as pd
import numpy as np
import holidays

file_path = '/Users/mymac/Google_Drive/Forex_Robot/sp500/'


# Get the data
def percentage_to_float(x):
  return float(x.strip('%')) / 100

us_holidays = holidays.US()

df = pd.read_csv(file_path + 'data/chris_full_data.csv')
df.columns = map(str.lower, df.columns)
df.drop(['simple date', 'day', 'day type name', 'holiday flag', 'date.1'], axis=1, inplace=True)
df.date = pd.to_datetime(df.date)
df.reset_index(drop=True, inplace=True)
df.fillna(0, inplace=True)
df['20 day rel vol %'] = df['20 day rel vol %'].astype(str)
df['20 day rel vol %'] = df['20 day rel vol %'].apply(lambda x: percentage_to_float(x))
daytype_onehot = pd.get_dummies(df.daytype, prefix='daytype')
df.drop('daytype', axis=1, inplace=True)
df = pd.concat([df, daytype_onehot], axis=1)
df['holiday'] = df['date'].apply(lambda x: 1 if x.strftime('%Y-%m-%d') in us_holidays else 0)

df


# Run simulation
reward = 0
n_wins = 0
n_losses = 0
win_streak = 0
loss_streak = 0
curr_win_streak = 0
curr_loss_streak = 0
n_buys = 0
n_sells = 0
trade = None
risk = 16 / 4
profit = 25 / 4

for i in range(df.shape[0]):
  on_open = df.loc[df.index[i], 'on open']
  tf_up = df.loc[df.index[i], '1tf up?']

  if tf_up == 1 and trade is None:
    open_price = on_open
    # stop_loss = on_open - risk
    # stop_gain = on_open + profit
    stop_loss = on_open + risk
    stop_gain = on_open - profit

    trade = {'open_price': open_price, 'stop_loss': stop_loss, 'stop_gain': stop_gain, 'trade_type': 'sell'}

  if trade is not None:
    on_high = df.loc[df.index[i], 'on hi.1']
    on_low = df.loc[df.index[i], 'on lo.1']
    on_close = df.loc[df.index[i], 'rth open.1']

    if trade['trade_type'] == 'buy':
      if on_low <= trade['stop_loss']:
        trade_amount = trade['stop_loss'] - trade['open_price']
        reward += trade_amount

        n_wins += 1 if trade_amount > 0 else 0
        n_losses += 1 if trade_amount < 0 else 0

        trade = None

        continue

      if on_high >= trade['stop_gain']:
        trade_amount = trade['stop_gain'] - trade['open_price']
        reward += trade_amount

        n_wins += 1 if trade_amount > 0 else 0
        n_losses += 1 if trade_amount < 0 else 0

        trade = None

        continue

    else:
      if on_high >= trade['stop_loss']:
        trade_amount = trade['open_price'] - trade['stop_loss']
        reward += trade_amount

        n_wins += 1 if trade_amount > 0 else 0
        n_losses += 1 if trade_amount < 0 else 0

        trade = None

        continue

      if on_low <= trade['stop_gain']:
        trade_amount = trade['open_price'] - trade['stop_gain']
        reward += trade_amount

        n_wins += 1 if trade_amount > 0 else 0
        n_losses += 1 if trade_amount < 0 else 0

        trade = None

        continue

print('Reward: ' + str(reward))
print('Wins: ' + str(n_wins))
print('Losses: ' + str(n_losses))
