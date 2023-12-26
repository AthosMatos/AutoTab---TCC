import numpy as np


def rmsNorm(S, level=-50.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = len(S) * r**2
    aDivide = np.sum(S**2)
    if aDivide == 0:
        return S
    a = np.sqrt(a / aDivide)
    # normalize
    S = S * a

    return S
