import numpy as np

def getMyPosition(prcSoFar):
    nInst, nDays = prcSoFar.shape
    if nDays < 15:
        return np.zeros(nInst)

    positions = np.zeros(nInst)

    # Settings
    lookback = 10     # days to find pivots
    trend_window = 6  # use last 6 days to detect pivot structure
    stop_pct = 0.05   # 2% stop-loss approximation
    dollar_per_signal = 1000  # how big each position is (keep it low to reduce cost)

    for i in range(nInst):
        price_series = prcSoFar[i, -lookback:]

        # Step 1: Detect local highs and lows (pivot points)
        highs = []
        lows = []

        for t in range(1, lookback - 1):
            if price_series[t] > price_series[t - 1] and price_series[t] > price_series[t + 1]:
                highs.append((t, price_series[t]))
            if price_series[t] < price_series[t - 1] and price_series[t] < price_series[t + 1]:
                lows.append((t, price_series[t]))

        # Step 2: Look at the last 2 pivot highs and lows
        if len(highs) >= 2 and len(lows) >= 2:
            prev_high = highs[-2][1]
            last_high = highs[-1][1]
            prev_low = lows[-2][1]
            last_low = lows[-1][1]

            # Step 3: Detect uptrend or downtrend
            uptrend = last_high > prev_high and last_low > prev_low
            downtrend = last_high < prev_high and last_low < prev_low

            min_trend_strength = 0.03  # % movement required
            high_change = (last_high - prev_high) / prev_high
            low_change = (last_low - prev_low) / prev_low

            if high_change < min_trend_strength or low_change < min_trend_strength:
                continue  # skip this instrument — trend not strong enough

            price_now = price_series[-1]

            # Step 4: Approximate stop-loss: if price dropped too much from recent high/low
            recent_max = np.max(price_series[-trend_window:])
            recent_min = np.min(price_series[-trend_window:])

            stop_long = price_now < recent_max * (1 - stop_pct)
            stop_short = price_now > recent_min * (1 + stop_pct)

            # Step 5: Decide position
            if uptrend and not stop_long:
                # Long signal
                dollar_target = dollar_per_signal
                positions[i] = int(dollar_target / price_now)
            elif downtrend and not stop_short:
                # Short signal
                dollar_target = dollar_per_signal
                positions[i] = int(-dollar_target / price_now)
            else:
                # No trend or stop condition met: stay flat
                positions[i] = 0