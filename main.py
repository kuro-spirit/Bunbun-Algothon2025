
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
prev_signal = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos, prev_signal
    (nins, nt) = prcSoFar.shape
    if nt < 51:
        return np.zeros(nins)

    returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])

    short_window = 5
    long_window = 50

    sma_short = np.mean(prcSoFar[:, -short_window:], axis=1)
    sma_long = np.mean(prcSoFar[:, -long_window:], axis=1)

    trend = sma_long - sma_short
    trend_signal = np.sign(trend)

    vol_long = np.std(returns[:, -long_window:], axis=1)
    vol_short = np.std(returns[:, -short_window:], axis=1)

    vol_ratio = vol_short / (vol_long + 1e-8)
    regime = vol_ratio > 1.5  # tighter filter

    raw_signal = trend_signal * regime
    raw_signal = np.where(np.abs(raw_signal) > 0.1, raw_signal, 0.0)

    # Smooth signal
    smoothed_signal = 0.7 * prev_signal + 0.3 * raw_signal
    prev_signal = smoothed_signal

    norm = np.sqrt(np.sum(smoothed_signal ** 2))
    if norm > 0:
        signal = smoothed_signal / norm
    else:
        signal = smoothed_signal

    dollar_budget_per_instrument = 800  # slightly lower size

    newPos = dollar_budget_per_instrument * signal / prcSoFar[:, -1]
    newPos = np.array([int(x) for x in newPos])

    # Reset position daily to reduce accumulation swings
    currentPos = newPos

    return currentPos
