import numpy as np
import pandas as pd

y = []
for i in range(3):
    print(i)
    y.append(i)

y_pd = pd.DataFrame(data=y)
y_pd.to_csv(r'stuff.csv', index=False, header=False)

y_readin = pd.read_csv('stuff.csv')
print(y_readin)
