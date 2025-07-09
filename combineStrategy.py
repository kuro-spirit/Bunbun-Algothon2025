import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 10):
        return np.zeros(nins)
    
    # === 1. Trend Signal: MA Crossover ===
    ma_short = np.mean(prcSoFar[:, -3:], axis=1)
    ma_long = np.mean(prcSoFar[:, -10:], axis=1)
    trend_signal = ma_short - ma_long

    # Normalize trend signal
    trend_signal -= np.mean(trend_signal)
    norm = np.linalg.norm(trend_signal)
    if norm > 1e-6:
        trend_signal /= norm
    else:
        return np.zeros(nInst)

    # === 2. Residual Mean Reversion Filter ===
    market = np.mean(prcSoFar, axis=0)         # Market = average of all prices
    market_change = market[-1] - market[-10]   # Change over the period

    # Estimate expected price change for each instrument
    expected_change = prcSoFar[:, -10] + market_change
    residual = prcSoFar[:, -1] - expected_change  # Actual - expected

    # Residual mean reversion signal (negative residual → undervalued → long)
    reversion_signal = -residual

    # Normalize
    reversion_signal -= np.mean(reversion_signal)
    norm = np.linalg.norm(reversion_signal)
    if norm > 1e-6:
        reversion_signal /= norm
    else:
        return np.zeros(nInst)

    # === Combine Signals ===
    combined_signal = 0.7 * trend_signal + 0.3 * reversion_signal

    # Keep only top-N strongest combined signals
    topN = 15
    strongest = np.argsort(-np.abs(combined_signal))[:topN]
    filtered_signal = np.zeros_like(combined_signal)
    filtered_signal[strongest] = combined_signal[strongest]
    signal = filtered_signal

    # Normalize final signal
    norm = np.linalg.norm(signal)
    if norm > 1e-6:
        signal /= norm
    else:
        return np.zeros(nInst)

    # === Convert to Positions ===
    prices_today = prcSoFar[:, -1]
    dollar_target = 150 * signal
    rpos = dollar_target / prices_today

    return rpos.astype(int)
