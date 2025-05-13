# -*- coding: utf-8 -*-
"""
Generating Fig.6
"""

import matplotlib.pyplot as plt

name = ['CVFS','FFS']
#t1= 26.864402294158936*0.00027778
t1=0.0075
#t2= 5966.97212600708*0.00027778
t2=1.6575
t = [t1,t2]

plt.figure(1)
bars=plt.bar(name, t, width=0.4,color='g',alpha=0.6)
plt.ylabel('hours')
plt.ylim(0,1.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')
    
plt.savefig("fig6.pdf")