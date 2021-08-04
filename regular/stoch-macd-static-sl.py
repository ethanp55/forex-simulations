import talib
import numpy as np
import pandas as pd

file_path = '/Users/mymac/Google_Drive/Forex_Robot/'


# ----------------------------------------------------------------------------------------------------
# Get the data
# ----------------------------------------------------------------------------------------------------
df = pd.read_csv(file_path + 'test1.csv')
df.Date = pd.to_datetime(df.Date)
df.reset_index(drop=True, inplace=True)


def add_fractal(df, i, look_back=3):
  if i >= look_back and i < df.shape[0] - look_back:
    lows = []
    highs = []

    for j in range(1, look_back + 1):
      prev_bid_low, prev_bid_high = df.loc[df.index[i - j], ['Mid_Low', 'Mid_High']]
      future_bid_low, future_bid_high = df.loc[df.index[i + j], ['Mid_Low', 'Mid_High']]

      lows.append(float(prev_bid_low))
      lows.append(float(future_bid_low))
      highs.append(float(prev_bid_high))
      highs.append(float(future_bid_high))

    bid_low, bid_high = df.loc[df.index[i], ['Mid_Low', 'Mid_High']]

    if float(bid_low) < min(lows):
      return 1

    elif float(bid_high) > max(highs):
      return 2

    else:
      return 0

  else:
    return np.nan


df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['Mid_Close'])
df['ema200'] = talib.EMA(df['Mid_Close'], timeperiod=200)
df['slowk'], df['slowd'] = talib.STOCH(df['Mid_High'], df['Mid_Low'], df['Mid_Close'])
df['atr'] = talib.ATR(df['Mid_High'], df['Mid_Low'], df['Mid_Close'], timeperiod=750)
df['fractal'] = [add_fractal(df, i) for i in range(df.shape[0])]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df['fractal'].value_counts()

value_per_pip = 1.0
max_pips_to_risk = 0.0100


# ----------------------------------------------------------------------------------------------------
# Get the dates
# ----------------------------------------------------------------------------------------------------
dates = df['Date']


# ----------------------------------------------------------------------------------------------------
# Simulation code
# ----------------------------------------------------------------------------------------------------
def run_simulation(risk_reward_ratio, pullback_cushion, time_window, atr_percentage):
    pullback_cushion /= 10000
    reward = 0
    n_wins = 0
    n_losses = 0
    win_streak = 0
    loss_streak = 0
    curr_win_streak = 0
    curr_loss_streak = 0
    n_buys = 0
    n_sells = 0
    length_divider = 24 * 12
    pips_risked = []
    day_fees = 0
    stoch_signal = None
    z = 0
    n_units = 50000
    trade = None

    for i in range(2, df.shape[0]):
        if z >= time_window:
            stoch_signal = None
            z = 0

        slowk1 = df.loc[df.index[i - 1], 'slowk']
        slowd1 = df.loc[df.index[i - 1], 'slowd']
        slowk2 = df.loc[df.index[i - 2], 'slowk']
        slowd2 = df.loc[df.index[i - 2], 'slowd']
        macd1 = df.loc[df.index[i - 1], 'macd']
        macdsignal1 = df.loc[df.index[i - 1], 'macdsignal']
        macd2 = df.loc[df.index[i - 2], 'macd']
        macdsignal2 = df.loc[df.index[i - 2], 'macdsignal']
        ema200 = df.loc[df.index[i - 1], 'ema200']
        bid_low1 = df.loc[df.index[i - 1], 'Mid_Low']
        bid_high1 = df.loc[df.index[i - 1], 'Mid_High']
        atr = df.loc[df.index[i - 1], 'atr']
        curr_ao = df.loc[df.index[i], 'Ask_Open']
        curr_bo = df.loc[df.index[i], 'Bid_Open']
        spread = abs(curr_ao - curr_bo)
        enough_volatility = (spread / atr) <= atr_percentage

        if slowk2 < slowd2 and slowk1 > slowd1 and max([20, slowk2, slowd2, slowk1, slowd1]) == 20:
            stoch_signal = 'buy'
            z = 0

        elif slowk2 > slowd2 and slowk1 < slowd1 and min([80, slowk2, slowd2, slowk1, slowd1]) == 80:
            stoch_signal = 'sell'
            z = 0

        if macd2 < macdsignal2 and macd1 > macdsignal1 and bid_low1 > ema200 and stoch_signal == 'buy' and enough_volatility and trade is None:
            pullback = None
            j = i - 4

            while j >= 0:
                curr_fractal = df.loc[df.index[j], 'fractal']

                if curr_fractal == 1:
                    # pullback = float(df_high.loc[df_high.index[j], 'Ask_Low'])
                    pullback = float(df.loc[df.index[j], 'Bid_Low'])
                    break

                j -= 1

            if pullback is not None:
                curr_bid_open, curr_bid_high, curr_bid_low, curr_bid_close, curr_ask_open, curr_ask_high, curr_ask_low, curr_ask_close = \
                df.loc[df.index[i], ['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High',
                                               'Ask_Low', 'Ask_Close']]
                open_price = float(curr_ask_open)
                stop_loss = round(pullback - pullback_cushion, 5)
                # stop_loss = round(pullback - pullback_cushion, 3)

                if stop_loss < open_price:
                    curr_pips_to_risk = open_price - stop_loss

                    if curr_pips_to_risk <= max_pips_to_risk:
                        stop_gain = round(open_price + (curr_pips_to_risk * risk_reward_ratio), 5)
                        # stop_gain = round(open_price + (curr_pips_to_risk * risk_reward_ratio), 3)

                        trade = {'open_price': open_price, 'trade_type': 'buy', 'stop_loss': stop_loss,
                                 'stop_gain': stop_gain, 'length': 0, 'pips_risked': round(curr_pips_to_risk, 5),
                                 'n_units': n_units}

                        n_buys += 1

                        pips_risked.append(curr_pips_to_risk)

                        stoch_signal = None

        elif macd2 > macdsignal2 and macd1 < macdsignal1 and bid_high1 < ema200 and stoch_signal == 'sell' and enough_volatility and trade is None:
            pullback = None
            j = i - 4

            while j >= 0:
                curr_fractal = df.loc[df.index[j], 'fractal']

                if curr_fractal == 2:
                    # pullback = float(df_high.loc[df_high.index[j], 'Bid_High'])
                    pullback = float(df.loc[df.index[j], 'Ask_High'])
                    break

                j -= 1

            if pullback is not None:
                curr_bid_open, curr_bid_high, curr_bid_low, curr_bid_close, curr_ask_open, curr_ask_high, curr_ask_low, curr_ask_close = \
                df.loc[df.index[i], ['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High',
                                               'Ask_Low', 'Ask_Close']]
                open_price = float(curr_bid_open)
                stop_loss = round(pullback + pullback_cushion, 5)
                # stop_loss = round(pullback + pullback_cushion, 3)

                if stop_loss > open_price:
                    curr_pips_to_risk = stop_loss - open_price

                    if curr_pips_to_risk <= max_pips_to_risk:
                        stop_gain = round(open_price - (curr_pips_to_risk * risk_reward_ratio), 5)
                        # stop_gain = round(open_price - (curr_pips_to_risk * risk_reward_ratio), 3)

                        trade = {'open_price': open_price, 'trade_type': 'sell', 'stop_loss': stop_loss,
                                 'stop_gain': stop_gain, 'length': 0, 'pips_risked': round(curr_pips_to_risk, 5),
                                 'n_units': n_units}

                        n_sells += 1

                        pips_risked.append(curr_pips_to_risk)

                        stoch_signal = None

        if stoch_signal is not None:
            z += 1

        if trade is not None:
            trade['length'] += 1

            curr_bid_open, curr_bid_high, curr_bid_low, curr_bid_close, curr_ask_open, curr_ask_high, curr_ask_low, curr_ask_close = \
            df.loc[
                df.index[i], ['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High', 'Ask_Low',
                                   'Ask_Close']]

            if trade['trade_type'] == 'buy' and curr_bid_low <= trade['stop_loss']:
                trade_amount = (trade['stop_loss'] - trade['open_price']) * trade['n_units'] * value_per_pip
                # trade_amount = (trade['stop_loss'] - trade['open_price']) * 100 * value_per_pip
                reward += trade_amount
                day_fees = day_fees + (-0.8 * int(trade['length'] / length_divider))

                n_wins += 1 if trade_amount > 0 else 0
                n_losses += 1 if trade_amount < 0 else 0
                curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                if curr_win_streak > win_streak:
                    win_streak = curr_win_streak

                if curr_loss_streak > loss_streak:
                    loss_streak = curr_loss_streak

                trade = None

                continue

            if trade['trade_type'] == 'buy' and curr_bid_high - trade['pips_risked'] > trade['stop_loss']:
                trade['stop_loss'] = curr_bid_high - trade['pips_risked']

            if trade['trade_type'] == 'buy' and curr_bid_high >= trade['stop_gain']:
                trade_amount = (trade['stop_gain'] - trade['open_price']) * trade['n_units'] * value_per_pip
                # trade_amount = (trade['stop_gain'] - trade['open_price']) * 100 * value_per_pip
                reward += trade_amount
                day_fees = day_fees + (-0.8 * int(trade['length'] / length_divider))

                n_wins += 1 if trade_amount > 0 else 0
                n_losses += 1 if trade_amount < 0 else 0
                curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                if curr_win_streak > win_streak:
                    win_streak = curr_win_streak

                if curr_loss_streak > loss_streak:
                    loss_streak = curr_loss_streak

                trade = None

                continue

            if trade['trade_type'] == 'sell' and curr_ask_high >= trade['stop_loss']:
                trade_amount = (trade['open_price'] - trade['stop_loss']) * trade['n_units'] * value_per_pip
                # trade_amount = (trade['open_price'] - trade['stop_loss']) * 100 * value_per_pip
                reward += trade_amount
                day_fees = day_fees + (-0.8 * int(trade['length'] / length_divider))

                n_wins += 1 if trade_amount > 0 else 0
                n_losses += 1 if trade_amount < 0 else 0
                curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                if curr_win_streak > win_streak:
                    win_streak = curr_win_streak

                if curr_loss_streak > loss_streak:
                    loss_streak = curr_loss_streak

                trade = None

                continue

            if trade['trade_type'] == 'sell' and trade['pips_risked'] + curr_ask_low < trade['stop_loss']:
                trade['stop_loss'] = trade['pips_risked'] + curr_ask_low

            if trade['trade_type'] == 'sell' and curr_ask_low <= trade['stop_gain']:
                trade_amount = (trade['open_price'] - trade['stop_gain']) * trade['n_units'] * value_per_pip
                # trade_amount = (trade['open_price'] - trade['stop_gain']) * 100 * value_per_pip
                reward += trade_amount
                day_fees = day_fees + (-0.8 * int(trade['length'] / length_divider))

                n_wins += 1 if trade_amount > 0 else 0
                n_losses += 1 if trade_amount < 0 else 0
                curr_win_streak = 0 if trade_amount < 0 else curr_win_streak + 1
                curr_loss_streak = 0 if trade_amount > 0 else curr_loss_streak + 1

                if curr_win_streak > win_streak:
                    win_streak = curr_win_streak

                if curr_loss_streak > loss_streak:
                    loss_streak = curr_loss_streak

                trade = None

                continue

    return reward + day_fees, n_buys, n_sells, n_wins, n_losses, win_streak, loss_streak


# ----------------------------------------------------------------------------------------------------
# Run simulation
# ----------------------------------------------------------------------------------------------------
# risk_reward_ratio_vals = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
risk_reward_ratio_vals = [1.5, 1.6, 1.7, 1.8, 1.9, 2]
# risk_reward_ratio_vals = [1.5]
# pullback_cushion_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# pullback_cushion_vals = [10, 15, 20, 25, 30, 35, 40, 45]
pullback_cushion_vals = [10, 15, 20, 25, 30, 35, 40]
# pullback_cushion_vals = [35]
time_windows = [10, 11]
# time_windows = [10]
atr_percentages = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
n_runs = len(risk_reward_ratio_vals) * len(pullback_cushion_vals) * len(time_windows) * len(atr_percentages)
best_risk_reward = None
best_pullback_cushion = None
best_time_window = None
best_atr_percentage = None
best_reward = -np.inf
runs_finished = 0

for risk_reward_ratio in risk_reward_ratio_vals:
  for pullback_cushion in pullback_cushion_vals:
    for time_window in time_windows:
        for atr_percentage in atr_percentages:
            reward, n_buys, n_sells, n_wins, n_losses, win_streak, loss_streak = run_simulation(risk_reward_ratio, pullback_cushion, time_window, atr_percentage)
            runs_finished += 1

            print(risk_reward_ratio)
            print(pullback_cushion)
            print(time_window)
            print(atr_percentage)
            print(reward)
            print('Num buys: ' + str(n_sells))
            print('Num sells: ' + str(n_buys))
            print('Num trades: ' + str(n_buys + n_sells))
            print('Num wins: ' + str(n_wins))
            print('Num losses: ' + str(n_losses))
            print('Win streak: ' + str(win_streak))
            print('Loss streak: ' + str(loss_streak))
            print('Remaining runs: ' + str(n_runs - runs_finished))

            if reward > best_reward:
              best_reward = reward
              best_risk_reward = risk_reward_ratio
              best_pullback_cushion = pullback_cushion
              best_time_window = time_window
              best_atr_percentage = atr_percentage

            print('Best reward so far: ' + str(best_reward))
            print()

print('------------ FINAL RESULTS ------------')
print('Best reward: ' + str(best_reward))
print('Best risk/reward ratio: ' + str(best_risk_reward))
print('Best pullback cushion: ' + str(best_pullback_cushion))
print('Best time window: ' + str(best_time_window))
print('Best atr percentage: ' + str(best_atr_percentage))

# TL:
# Best reward: 3097.400000000116
# Best risk/reward ratio: 1.6
# Best pullback cushion: 35
# Best time window: 10
# Best atr percentage: 0.35
