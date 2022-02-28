import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl

import helperFunctions as f
from math import sqrt,pi
from schmidtForm import schmidt

print("Running")
def discretizeFunc(func,numSamples,bandwidth,tpe='lin',logFactor=1.2,b=0.25):
    '''DEPRECIATED'''
    if tpe=='lin':
        dx = 2*bandwidth/numSamples
        bins = [-bandwidth+dx*n for n in range(0,numSamples+1)]
    elif tpe=='log':
        N=(numSamples+2)/2
        N = int(N)
        bins = [logFactor**-m for m in range(N)]
        bins2 = bins.copy()
        bins = [-x for x in bins2]
        bins2.reverse()
        bins = bins+bins2
    elif tpe=='linlog':
        n=int(numSamples/2)
        N=(numSamples-n+2)/2
        N = int(N)
        bins = [logFactor**-m for m in range(N)]
        bins2 = bins.copy()
        bins = [-x for x in bins2]
        bins2.reverse()
        dx = (bins[-1]-bins2[0])/n
        dx = abs(dx)
        bins3 = [bins[-1]+dx*k for k in range(1,n)]
        bins = bins+bins3+bins2
    elif tpe == 'linlog_inverted':
        #There are 46 modes per bath at N=92 bath sites
        n = int(numSamples*0.25/2)     #How many modes to put in the logarithmic part
        N = int(numSamples/2 - n)+1          #How many modes to put in the linear part
        print(n,N)
        #b=0.25
        shift = -0.5*(logFactor+1)*(logFactor)**-n
        bins_log = [b*(logFactor)**-m +b*shift for m in range(n)]
        bins_log.reverse()
        dx_lin = (bandwidth-b)/N
        bins_lin = [b+dx_lin*k for k in range(1,N+1)]
        bins = (bins_log+bins_lin).copy()
        bins.reverse()
        bins.extend([-a for a in bins_log+bins_lin])
        bins.reverse()
    samples = [sp.integrate.quad(func,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = [sp.integrate.quad(lambda x:func(x)*x,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = map(lambda x,y:x/y,energies,samples)
    samples = map(sqrt,samples)
    return list(energies),list(samples)

nBath = 99                            #In this version of the code, choose nBath=odd so nTotal = even for convenience
N_up = int(sp.ceil((nBath+1)*0.5))    #Number of one spin flavour
N_dn = int(sp.floor((nBath+1)*0.5))   #Number of the other spin flavour         
#-----|Algorithm parameters|-----
cTol = 10**-8                         #Tolerance for determining whether a state contributes to the ground state. In this version, this is the bare amplitude
#------|Other convenient parameters|-----
N_tot = N_up + N_dn                            #Total number of particles
nSites = nBath +1                              #Total number of sites
sites = sp.linspace(0,nSites-1,nSites)         #Site indices
binomials = f.binomial(nSites+1)               #Compute binomial coefficients for bitstring hash functions
ePrev = 10.0                                   #Monitor convergence of ground state. Initialized to random large values

#For now, we only work at half-filling
if nSites%2 !=0:
    raise Exception('The number of total sites should be even. The total number of sites is set to {}'.format(nSites))

#Initialize bath - flat DoS
D = 1
couplingStrength = 0.1

#Initialize bath - SE DoS
#DoS = lambda x: 2*couplingStrength*sqrt(1-x**2)/pi

#Bath - Flat DoS
DoS = lambda x: couplingStrength/sqrt(nBath)
Eb,Vb = discretizeFunc(DoS,nBath,D,tpe='linlog_inverted',logFactor=1.5,b=0.1)
Delta = pi*sum([x**2 for x in Vb])/2 #Multiply by value of DoS at Ef. Spectral function is then pinned to 1/pi*delta
H_bath = sp.diag(Eb)
H_1p = sp.linalg.block_diag([0],H_bath)
H_1p[0,1:1+len(Eb)] = Vb
H_1p[1:1+len(Eb),0] = Vb

U = 4*pi*Delta                #Hubbard interaction
zvals = [1-0.05*n for n in range(20)]                 #Rescaling parameter for hybridization
basis_size =  []              #Record basis size and energy for different z
E = []
gamma = []
a1 = []
a2 = []
b1 = []
b2 = []
dE = []
overlap = []
zvals.reverse()
for i, z in enumerate(zvals):
    #Calculate density matrices from which we wish to obtain a Schmidt basis
    H_bath = sp.diag(Eb)
    H_r = sp.linalg.block_diag([0],H_bath)
    H_r[0,1:1+len(Eb)] = [v*z for v in Vb]
    H_r[1:1+len(Eb),0] = [v*z for v in Vb]
    (en,eV) = sp.linalg.eigh(H_r)
    sort = sp.argsort(en)
    eV=eV[:,sort]
    dens_matrix = sp.matmul(eV[:,0:N_up],eV[:,0:N_dn].transpose())
    en,ev = sp.linalg.eigh(dens_matrix)
    U_s = schmidt(H_r,dens_matrix,1)
    H_s = U_s.transpose()*H_1p*U_s
    gamma.append(H_s[0,int((nBath+1)/2)]**2)
    a1.append(H_s[0,1]**2)
    a2.append(H_s[0,1+int((nBath+1)/2)]**2)
    b1.append(H_s[1,int((nBath+1)/2)]**2)
    b2.append(H_s[int((nBath+1)/2),1+int((nBath+1)/2)]**2)
    dE.append(abs(H_s[1,1]-H_s[1+int((nBath+1)/2),1+int((nBath+1)/2)]))
    if z==1:
        E = [H_s[k,k] for k in range(1,int((nBath+1)/2))]
        E2 = [H_s[k,k] for k in range(int((nBath+1)/2)+1,nBath+1)]
    if i == 0:
        #partial = U_s[1+int(nBath/2),:]
        partial = U_s[:,1+int(nBath/2)]
    overlap.append(abs(float(U_s[:,1+int(nBath/2)].transpose()*partial)))
#partial = U_s[1+int((nBath+1)/2),:]
#40 sites, so 19 sites in each chain. Impurity located at pos. 20, partially occupied site at pos. 41
#print(gamma[0]+a1[0]+a2[0],2*Delta/pi)
figs,ax = mpl.subplots(2,2)
ax[0,0].plot(zvals,gamma,'x')
ax[0,0].plot(zvals,a1,'ro')
ax[0,0].plot(zvals,b1,'bo')
ax[0,0].plot(zvals,b2,'kx')
ax[0,0].legend(['gamma','a1','b1','b2'])
ax[1,0].plot(zvals,dE,'x')
ax[0,1].plot(E,'x')
ax[0,1].plot(E2,'o')
ax[1,1].plot(zvals,overlap,'o')

mpl.show()