

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