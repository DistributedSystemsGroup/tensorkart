import numpy as np

values = np.loadtxt( 'all_data/luigi_raceway1/data.csv', delimiter=',', usecols=(1,3))
print(values)

