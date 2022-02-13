import matplotlib.pyplot as pyp
import numpy as np

data = np.load('results_D=10_c=0.1_N=100.npz')
pyp.plot(data['Uvals'],data['R'],'x')
pyp.show()