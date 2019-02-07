import numpy as np
from sklearn.decomposition import PCA

a = np.array([[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]])

print(a)


pca = PCA(1, svd_solver='full')
pca.fit(a)
print('++++')
print(a)
pc = pca.components_
# projected values on the principal component
T_a = np.matmul(a, pc.transpose(1, 0))
print('++++')
print(T_a)

t2 = pca.fit_transform(a)
print('++++')
print(t2)
