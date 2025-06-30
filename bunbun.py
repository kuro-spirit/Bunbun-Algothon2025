
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 11):
        return np.zeros(nins)
    
    # 5-day momentum
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -11]) # computes log return of each instrument from yesterday to today

    # lNorm = np.sqrt(lastRet.dot(lastRet)) # calculates l2 norm of return vector

    returns = np.diff(np.log(prcSoFar), axis=1)  # daily log returns
    vol = np.std(returns[:, -10:], axis=1) + 1e-8  # recent 5-day vol per instrument

    # Risk-adjusted signal
    signal = lastRet / vol
    signal[np.abs(signal) < 0.2] = 0

    # Normalize signal
    norm = np.linalg.norm(signal)
    if norm > 1e-6:
        signal = signal / norm
    else:
        return np.zeros(nInst)  # do nothing if signal is zero

    dollar_target = 750 * signal
    rpos = np.array([int(x) for x in dollar_target / prcSoFar[:, -1]]) # converts signal into target positions (get number of shares)
    # currentPos = np.array([int(x) for x in currentPos+rpos])
    top5 = np.argsort(-np.abs(signal))[:5]

    return rpos
