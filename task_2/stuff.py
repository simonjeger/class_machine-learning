import numpy as np
import pandas as pd

s = pd.Series([10, 20, 30])
print(s)
labels = [1, 2, 3]
print(s.loc[s.index.intersection(labels)])
