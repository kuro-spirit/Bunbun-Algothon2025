import numpy as np

def getMyPosition(prcSoFar):
    nInst, nDays = prcSoFar.shape
    if nDays < 11:
        return np.zeros(nInst)
    
    # 5-day momentum
    momentum = np.log(prcSoFar[:, -1] / prcSoFar[:, -6])
    # Rank-based momentum (centered)
    ranks = np.argsort(np.argsort(momentum))
    rank_signal = ranks - np.mean(ranks)


    log_returns = np.diff(np.log(prcSoFar[:, -16:]), axis=1)    # Compute daily log returns for last 11 days
    today_ret = log_returns[:, -1]     # Use last day's return as signal
    # Use std dev of previous 10 days to assess significance
    vol = np.std(log_returns[:, :-1], axis=1) + 1e-8  # avoid divide by zero
    zscore = today_ret / vol     # Compute z-score of today's move

    # Use only instruments with strong breakout
    signal = rank_signal * np.where(np.abs(zscore) > 3.0, 1, 0)  # only act if breakout is strong

    # Keep only top 10 breakout
    topN = 10
    strongest = np.argsort(-np.abs(signal))[:topN]
    filtered_signal = np.zeros_like(signal)
    filtered_signal[strongest] = signal[strongest]
    signal = filtered_signal

    # Normalize
    norm = np.linalg.norm(signal)
    if norm > 1e-6:
        signal /= norm
    else:
        return np.zeros(nInst)

    # Position sizing
    prices_today = prcSoFar[:, -1]
    dollar_target = 150 * signal
    rpos = dollar_target / prices_today

    return rpos.astype(int)