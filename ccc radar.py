import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

N=16
theta=0.1
L=8
PTM = N/2

#golayの自己相関関数
x = np.array([1,1,-1,1,1,1,1,-1])
y = np.array([-1,-1,1,-1,1,1,1,-1])

corrx = sp.correlate(x, x, mode='full')/L
corry = sp.correlate(y, y, mode='full')/L

ptm = np.array([1,1,0,1,0,0,1])

corr_golay = np.array([corrx,corry])

for flag in ptm:
    if flag==0:
        corr_golay = np.append(corr_golay,[corrx],axis=0)
        corr_golay = np.append(corr_golay,[corry],axis=0)
    else:
        corr_golay = np.append(corr_golay,[corry],axis=0)
        corr_golay = np.append(corr_golay,[corrx],axis=0)


corr_conv = np.array([corrx,corry]*8)
#print(corr_golay)

amb_conv = np.zeros(N-1,dtype=np.complex)
amb_golay= np.zeros(N-1,dtype=np.complex)

for i in range(N):
    amb_conv +=  corr_conv[i]*np.exp(complex(0,i)*theta)
    amb_golay  +=  corr_golay[i]*np.exp(complex(0,i)*theta)


x = np.arange(0,15,1)
y_conv= np.abs(amb_conv[x])
y_golay = np.abs(amb_golay[x])

fig = plt.figure(figsize=(8, 8))
plt.plot(x,y_conv,label="conventional")
plt.plot(x,y_golay,label="golay")

plt.legend()
plt.savefig("golay_singlech.png")
plt.show()
