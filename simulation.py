import holidays
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

file_path = '/Users/mymac/Google_Drive/Forex_Robot/sp500/'


# ----------------------------------------------------------------------------------------------------
# Get the data
# ----------------------------------------------------------------------------------------------------
def percentage_to_float(x):
  return float(x.strip('%')) / 100

us_holidays = holidays.US()

df = pd.read_csv(file_path + 'data/full-data.csv')
df.columns = map(str.lower, df.columns)
df.drop(['simple date', 'day', 'day type name', 'holiday flag', 'date.1'], axis=1, inplace=True)
df.date = pd.to_datetime(df.date)
df[['beob', 'buob', 'bull hammer', 'bear hammer', 'bullish idf', 'bearish idf', 'bull pinbar', 'bear pinbar']] = df[['beob', 'buob', 'bull hammer', 'bear hammer', 'bullish idf', 'bearish idf', 'bull pinbar', 'bear pinbar']].fillna(0)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['% distance of 5ema to 50ema'] = df['% distance of 5ema to 50ema'].astype(str)
df['% distance of 5ema to 50ema'] = df['% distance of 5ema to 50ema'].apply(lambda x: percentage_to_float(x))
df['% distance of 5ema to 100ema'] = df['% distance of 5ema to 100ema'].astype(str)
df['% distance of 5ema to 100ema'] = df['% distance of 5ema to 100ema'].apply(lambda x: percentage_to_float(x))
df['% distance of 50ema to 100ema'] = df['% distance of 50ema to 100ema'].astype(str)
df['% distance of 50ema to 100ema'] = df['% distance of 50ema to 100ema'].apply(lambda x: percentage_to_float(x))
df['% gain on the day'] = df['% gain on the day'].astype(str)
df['% gain on the day'] = df['% gain on the day'].apply(lambda x: percentage_to_float(x))
df['range as a % of price'] = df['range as a % of price'].astype(str)
df['range as a % of price'] = df['range as a % of price'].apply(lambda x: percentage_to_float(x))
df['20 day rel vol %'] = df['20 day rel vol %'].astype(str)
df['20 day rel vol %'] = df['20 day rel vol %'].apply(lambda x: percentage_to_float(x))
daytype_onehot = pd.get_dummies(df.daytype, prefix='daytype')
df.drop('daytype', axis=1, inplace=True)
df = pd.concat([df, daytype_onehot], axis=1)
df['holiday'] = df['date'].apply(lambda x: 1 if x.strftime('%Y-%m-%d') in us_holidays else 0)
df['sin_day'] = np.sin(2 * np.pi * df['date'].dt.day / 7)
df['cos_day'] = np.cos(2 * np.pi * df['date'].dt.day / 7)
df['sin_month'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['cos_month'] = np.cos(2 * np.pi * df['date'].dt.month / 12)


# ----------------------------------------------------------------------------------------------------
# Create targets
# ----------------------------------------------------------------------------------------------------
account_balance = 10000
value_per_tick = 12.5
ticks_per_num = 4
max_balance_to_risk = account_balance * 0.02
risk_reward_ratio = 1.5
stop_loss = (max_balance_to_risk / value_per_tick) / ticks_per_num

def create_target(dataset, i, pred_period=1):
  tmp = i
  j = 0
  buy = False
  sell = False

  # Assume a buy
  open_price = dataset.loc[dataset.index[i], ['open']]
  buy_open_price = float(open_price)
  buy_stop_loss = buy_open_price - stop_loss
  buy_stop_gain = buy_open_price + (risk_reward_ratio * stop_loss)

  while i < dataset.shape[0] and j < pred_period:
    curr_high = dataset.loc[dataset.index[i], 'on hi']
    curr_low = dataset.loc[dataset.index[i], 'on lo']

    if curr_low <= buy_stop_loss:
      loss_amount = (buy_stop_loss - buy_open_price) * 4 * value_per_tick

      if loss_amount > 0:
        buy = True

      break

    if curr_high >= buy_stop_gain:
      buy = True
      break

    i += 1
    j += 1

  i = tmp
  j = 0

  # Assume a sell
  open_price = dataset.loc[dataset.index[i], ['open']]
  sell_open_price = float(open_price)
  sell_stop_loss = sell_open_price + stop_loss
  sell_stop_gain = sell_open_price - (risk_reward_ratio * stop_loss)

  while i < dataset.shape[0] and j < pred_period:
    curr_high = dataset.loc[dataset.index[i], 'on hi']
    curr_low = dataset.loc[dataset.index[i], 'on lo']

    if curr_high >= sell_stop_loss:
      loss_amount = (sell_open_price - sell_stop_loss) * 4 * value_per_tick

      if loss_amount > 0:
        sell = True

      break

    if curr_low <= sell_stop_gain:
      sell = True
      break

    i += 1
    j += 1

  if buy and sell:
    return 0

  elif buy:
    return 1

  elif sell:
    return 2

  else:
    return 0


targets = [create_target(df, i) for i in range(df.shape[0] - 1)]
targets.append(np.nan)
df['target'] = targets
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df['target'].value_counts())


# ----------------------------------------------------------------------------------------------------
# Get the dates
# ----------------------------------------------------------------------------------------------------
dates = df['date']
# dates[576]
dates[872]
df.drop('date', axis=1, inplace=True)

dates = dates.iloc[872:,]
dates.reset_index(drop=True, inplace=True)
df = df.iloc[872:,]
df.reset_index(drop=True, inplace=True)


# ----------------------------------------------------------------------------------------------------
# Create x and y data sets for training and testing
# ----------------------------------------------------------------------------------------------------
x = df.loc[:, df.columns != 'target']
y = df['target']

test_percentage = 0.3
cutoff_index = int(len(df) * (1 - test_percentage))

x_train_org = x.iloc[0:cutoff_index, :]
x_test_org = x.iloc[cutoff_index:, :]
y_train = y.iloc[0:cutoff_index, ]
y_test = y.iloc[cutoff_index:, ]

# Scale the input data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_org)
x_test = scaler.transform(x_test_org)


# ----------------------------------------------------------------------------------------------------
# Train the svm
# ----------------------------------------------------------------------------------------------------
svm = SVC(kernel='linear', gamma=10, class_weight='balanced', C=1000)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Balanced accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))


# ----------------------------------------------------------------------------------------------------
# Get the 15 minute data
# ----------------------------------------------------------------------------------------------------
# df_15 = pd.read_csv(file_path + 'data/sp500-15m.csv')
df_15 = pd.read_csv(file_path + 'data/sp500-5m.txt', delimiter='\t')
df_15.columns = map(str.lower, df_15.columns)
df_15.rename(columns={'date time': 'date', 'last': 'close'}, inplace=True)
# df_15.drop(['name', 'volume', 'open interest'], axis=1, inplace=True)
df_15.date = pd.to_datetime(df_15.date)


# ----------------------------------------------------------------------------------------------------
# Make the dates vector the same length as x_test (so we can keep track of what date we're on)
# ----------------------------------------------------------------------------------------------------
dates = np.expand_dims(dates[cutoff_index:], axis=1)


# ----------------------------------------------------------------------------------------------------
# Run the simulation
# ----------------------------------------------------------------------------------------------------
i = 0
n_buys = 0
n_sells = 0
reward = 0
n_wins = 0
n_losses = 0


def iterate_thru_15m_data(trade, start_date):
  def return_next_date(j):
    return df_15.loc[df_15.index[j + 1], 'date'] if j + 1 < len(df_15) else -1

  z = 0
  on_length = 60 # ON session is about 15 hours -> 15 * 4 = 60

  j = df_15[df_15.date >= start_date].index[0]
  open_price = float(df_15.loc[df_15.index[j], 'open'])

  if trade == 'buy':
    # trade_stop_loss = open_price - stop_loss
    # trade_stop_gain = open_price + (risk_reward_ratio * stop_loss)
    trade_stop_loss = open_price - 7.5
    trade_stop_gain = open_price + 12.5

  else:
    trade_stop_loss = open_price + stop_loss
    trade_stop_gain = open_price - (risk_reward_ratio * stop_loss)

  global reward
  global n_losses
  global n_wins

  while j < len(df_15):
    curr_high, curr_low, curr_open = df_15.loc[df_15.index[j], ['high', 'low', 'open']]

    # If it's a buy
    if trade == 'buy':
      if curr_low <= trade_stop_loss:
        ticks_lost = (trade_stop_loss - open_price) * 4
        reward += ticks_lost
        n_losses += 1

        return return_next_date(j)

      # if z >= on_length:
      #   ticks = (curr_open - open_price) * 4
      #   reward += ticks
      #   n_wins += 1 if ticks > 0 else 0
      #   n_losses += 1 if ticks < 0 else 0
      #
      #   return return_next_date(j)

      if curr_high >= trade_stop_gain:
        ticks_gained = (trade_stop_gain - open_price) * 4
        reward += ticks_gained
        n_wins += 1

        return return_next_date(j)

    # If it's a sell
    else:
      if curr_high >= trade_stop_loss:
        ticks_lost = (open_price - trade_stop_loss) * 4
        reward += ticks_lost
        n_losses += 1

        return return_next_date(j)

      if curr_low <= trade_stop_gain:
        ticks_gained = (open_price - trade_stop_gain) * 4
        reward += ticks_gained
        n_wins += 1

        return return_next_date(j)

    j += 1
    z += 1

# while i < len(x_test):
while i < len(df):
  def get_new_i(stop_date):
    new_i = np.where(dates >= stop_date)[0]

    if len(new_i) > 0:
      return new_i[0]

    else:
      return len(df) + 50

  # curr_date = dates[i][0]
  curr_date = dates[i]
  # curr_row = x_test[i, :].reshape(1, -1)
  vix, range_percentage, close_up = df.loc[df.index[i], ['vix adj close', 'range as a % of price', 'on close up?']]
  # curr_pred = svm.predict(curr_row)[0]
  curr_pred = vix >= 18 and range_percentage >= 0.01 and close_up == 1

  # if curr_pred != 0:
  if curr_pred:
    # trade = 'buy' if curr_pred == 1 else 'sell'
    trade = 'buy'
    n_buys += 1 if trade == 'buy' else 0
    n_sells += 1 if trade == 'sell' else 0
    stop_date = iterate_thru_15m_data(trade, curr_date)

    if stop_date == -1:
      # i = len(x_test) + 50
      i = len(df) + 50

    else:
      i = get_new_i(stop_date)

  else:
    i += 1

print('Start date: ', dates[0])
print('Total buys: ', n_buys)
print('Total sells: ', n_sells)
print('Total trades: ', n_buys + n_sells)
print('Total wins: ', n_wins)
print('Total losses: ', n_losses)
print('Total ticks gained: ', reward)
