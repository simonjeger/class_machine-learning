import numpy as np
import pandas as pd

def create_false_triplets():
    y_readin = pd.read_csv('../data_4/my_triplets_truelabel.txt', delimiter=' ', header=None)
    y_readin = np.asarray(y_readin)
    y = np.c_[y_readin[:,0], y_readin[:,2], y_readin[:,1]]


    y_pd = pd.DataFrame(data=y)
    y_pd.to_csv(r'../data_4/my_triplets_falselabel.txt', index=False, header=False)

    print(y_readin)
    return


### MAIN
#create_false_triplets()

y_readin = pd.read_csv('hyperparam_settings.csv', delimiter=',')
y_sorted = y_readin.sort_values(by=['accuracy'], ascending=False)
print(y_sorted)
