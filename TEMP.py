import datetime as dt
import numpy as np

t = '20220611'
print(np.datetime64(t))
print(dt.datetime.strptime(t, "%Y%m%d").date())