

# A plotting function for how I defined the error

import numpy as np
import matplotlib.pyplot as plt

data = np.load('errors.npz')
M1 = data['M1error']
M2 = data['M2error']
M3 = data['M3error']


plt.hist(M1,15,edgecolor='k', alpha=0.5, label='Simple' )
plt.hist(M2,15,edgecolor='k', alpha=0.5, label='Deep' )
plt.hist(M3,15,edgecolor='k', alpha=0.5, label='Wide' )
plt.title('tSNE: NN Loss', fontweight='bold', fontsize=16)
plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.legend(loc='upper right')
plt.savefig('tSNE_Model.svg', format='svg')

plt.show()



plt.savefig('tSNE_Model.svg', format='svg')


###################################################################
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
data = np.load('data/data_w_labels.npz')
Bdata = data['vec']     # Binder Word Vectors
Gdata = data['gVec']    # Google word Vectors
L1 = data['L1']     # Super Category labels
L2 = data['L2']     # Category labels
# New Distances, but real
nnB = np.zeros((534,534))
nnG = np.zeros((534,534))
for row in range(0,534):
    for column in range(0,534):
        #print( row, ' ' ,column)
        nnB[row, column] = np.linalg.norm(Bdata[row,:] - Bdata[column,:])
        nnG[row, column] = np.linalg.norm(Gdata[row,:] - Gdata[column,:])