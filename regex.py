import re
import numpy as np
def remove_n_space(s):
    try:
        new = re.sub('([\n])', '',s)
    except TypeError:
        new = np.nan
    except ValueError:
        new = np.nan
    return new

def remove_white_space(s):
    try:
        new = int(re.sub('([\s])', '',s))
    except TypeError:
        new = np.nan
    except ValueError:
        new = np.nan
    return new