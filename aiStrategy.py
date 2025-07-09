import numpy as np

def getMyPosition(prcSoFar):
    nInst, nDays = prcSoFar.shape
    positions = np.zeros(nInst)

    # === Fixed Strategy Hyperparameters (Tunable Manually) ===
    short_ma_window = 4
    long_ma_window = 12
    vol_window = 7
    vol_mult = 0.6
    base_dollar_per_signal = 25

    max_position = 2

    # === Alpha-Based High Score Strategy ===
    if nDays < max(short_ma_window, long_ma_window, vol_window) + 1:
        return positions  # Not enough data yet

    for i in range(nInst):
        price_series = prcSoFar[i]
        price_now = price_series[-1]

        short_ma = np.mean(price_series[-short_ma_window:])
        long_ma = np.mean(price_series[-long_ma_window:])
        recent_volatility = np.max(price_series[-vol_window:]) - np.min(price_series[-vol_window:])

        if recent_volatility == 0:
            continue  # Skip flat instruments

        signal_strength = abs(price_now - short_ma) + abs(price_now - long_ma)

        if signal_strength > vol_mult * recent_volatility:
            direction = -1 if price_now > short_ma and price_now > long_ma else 1
            dollar_per_signal_scaled = base_dollar_per_signal * max_position * (signal_strength / recent_volatility)
            position = int(dollar_per_signal_scaled / price_now)
            positions[i] = direction * position

    return positions
