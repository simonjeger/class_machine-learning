import pandas as pd

M_Sub_panda = pd.read_csv('../task_2/sample.csv', delimiter=',')

compression_opts = dict(method='zip', archive_name='sample.csv')
M_Sub_panda.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
