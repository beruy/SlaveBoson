import numpy as np
import scipy as sp
from scipy.linalg import pinv as inverse
import scipy.linalg
from math import pi,sqrt
import matplotlib.pyplot as py
from functools import partial

#R comes with a 1/sqrt(2) (I think)?

def globalNewton(funcs,b,J,tol=10**-4,lmin=10**-1,halt=False):
    'Globally convergent Newton algorithm'
    def X(x):
        'Converts a matrix into a numpy array with boson labels'
        temp = np.array([0+0*1j],dtype=[(amp,'complex64') for amp in ('D','E','up','dn','Dstar','Estar','upstar','dnstar','l0','lup','ldn')])
        for i,bos in enumerate(['l0','lup','ldn','E','D','up','dn','Estar','Dstar','upstar','dnstar']):
            temp[bos] = x[i]
        return temp

    #1.) Compute Newton direction
    mixPrev = 1
    mix = 1
    g = lambda funcs,b: 0.5*sp.linalg.norm([f(b) for f in funcs])**2
    dg = lambda df,dx: (df*dx)[0,0]
    x = sp.matrix([complex(b[bos]) for bos in ('l0','lup','ldn','E','D','up','dn','Estar','Dstar','upstar','dnstar')])
    F = sp.matrix([f(b) for f in funcs]).transpose()
    xnew = -inverse(J)*F
    g1 = g(funcs,X(x.transpose()+xnew))
    g0 = g(funcs,b)
    #dF = sp.matrix([f(X(x.transpose()+xnew))-f(b) for f in funcs])
    dF = np.conj(F).transpose()*J            #Do we need F conjugate?? Then we are minimizing conj(F) F which is presubably the proper measure
    itern = 0
    #print(g1 - g0 - tol*dg(dF,xnew))
    while g1 - g0 - tol*dg(dF,xnew)>0 and itern<10:
        #print("Iteration number {}".format(itern))
        #print("g0 = {}".format(g0))
        if itern == 0:
            mix = np.real(-0.5*dg(dF,xnew)/(g1-g0-dg(dF,xnew)))
            #print("Mixing parameter = {}".format(mix))
            '''
            if mix<0.1:
                print(g1)
                gvals = [g(funcs,X(x.transpose()+l0*0.3*xnew/50)) for l0 in range(50)]
                py.plot([0.3*k/50 for k in range(50)],gvals,'o')
                py.hlines(g0,0,0.3)
                py.show()
            '''
            if mix<lmin:
                mix = lmin
            g1 = g(funcs,X(x.transpose()+mix*xnew)) 
            #print("g_mid = {}".format(g1))
            itern+=1
        else:
            g2 = g(funcs,X(x.transpose()+mixPrev*xnew))
            m1 = sp.matrix([[1/(mix)**2,-1/(mixPrev)**2],[-mixPrev/(mix)**2,mix/(mixPrev)**2]])/(mix-mixPrev)
            m2 = sp.matrix([g1-dg(dF,xnew)*mix-g0,g2-dg(dF,xnew)*mixPrev-g0]).transpose()
            params = m1*m2
            radical = np.abs((params[1]**2)-3*params[0]*dg(dF,xnew))
            mixPrev = mix
            mix = ((-params[1]+sqrt(radical))/(3*params[0]))[0,0]
            #print("Refined mixing parameter = {}".format(mix))
            g1 = g(funcs,X(x.transpose()+mix*xnew))
            #print("g_mid = {}".format(g1))
            itern+=1
            #quit()
    sizeofCorr = sp.real(sp.linalg.norm(xnew)*mix)
    if sizeofCorr>1:
        mix = 0.1
        #raise Exception("Newton step divergent")
    #print("Size of correction vector: {}".format(sizeofCorr))
    if mix<lmin:
        mix = lmin
    return X(x.transpose()+mix*xnew), float(sizeofCorr)
def dM(b,spin,bos):
    #Must change function calls as I removed M from the argument
    k1 = np.real(b[Conj['D']]*b['D'] + b[Conj[spin]]*b[spin])
    k2 = np.real(b[Conj['E']]*b['E'] + b[Conj[Not[spin]]]*b[Not[spin]])
    m = 1/sqrt(k1*k2)
    if bos in ('E','Estar',Not[spin],Conj[Not[spin]]):
        return complex(-0.5*b[Conj[bos]]*m/k2)
    elif bos in ('D','Dstar',Conj[spin],spin):
        return complex(-0.5*b[Conj[bos]]*m/k1)

def d2m(b,spin,bos,bos2):
    k1 = np.real(b[Conj['D']]*b['D'] + b[Conj[spin]]*b[spin])
    k2 = np.real(b[Conj['E']]*b['E'] + b[Conj[Not[spin]]]*b[Not[spin]])
    #print(k1,k2)
    A = ['D',spin,'Dstar',Conj[spin]]
    B = ['E',Not[spin],'Estar',Conj[Not[spin]]]
    res = 0+0*1j
    m = 1/sqrt(k1*k2)
    #First term
    if bos in B:
        res += complex(-0.5*b[Conj[bos]]*dM(b,spin,bos2)/k2)
    else:
        res += complex(-0.5*b[Conj[bos]]*dM(b,spin,bos2)/k1)
    #Second term
    if bos in A and bos2 in A:
        res += complex(+0.5*b[Conj[bos]]*b[Conj[bos2]]*m/k1**2)
        if bos is Conj[bos2]:
            res += complex(-0.5*m/k1)
    elif bos in B and bos2 in B:
        res += complex(0.5*b[Conj[bos]]*b[Conj[bos2]]*m/k2**2)
        if bos is Conj[bos2]:
            res += complex(-0.5*m/k2)
    return res

def dR(b,spin,bos,conj=False):    
    A = ['D','Estar',spin,Conj[Not[spin]]]
    C = [Conj[Not[spin]],spin,'Estar','D']
    if conj is False:
        if bos in A:
            return complex(b[C[A.index(bos)]])
        else:
            return 0 
    else:
        if Conj[bos] in A:
            return complex(b[Conj[C[A.index(Conj[bos])]]])
        else:
            return 0

def d2r(spin,bos,bos2,conj=False):
    A = ['D','Estar',spin,Conj[Not[spin]]]
    C = [Conj[Not[spin]],spin,'Estar','D']
    if conj is False:
        if (bos,bos2) in zip(A,C):
            return 1
        else:
            return 0
    elif conj is True:
        if (Conj[bos],Conj[bos2]) in zip(A,C):
            return 1
        else:
            return 0

def pref(b,bos):
    if bos in ('E','Estar'):
        return complex(b[Conj[bos]]*(E[bos]-2*pi*b['l0']))
    elif bos in ('D','Dstar'):
        return complex(b[Conj[bos]]*(E[bos]-2*pi*sum(b[m] for m in ('l0','lup','ldn'))))
    elif bos in ('up','dn'):
        return complex(b[Conj[bos]]*(E[bos]-2*pi*b['l0'] - 2*pi*b['l'+bos]))
    elif bos in ('upstar','dnstar'):
        return complex(b[Conj[bos]]*(E[bos]-2*pi*b['l0'] - 2*pi*b['l'+Conj[bos]]))
def dPref(b,bos,bos2):
    if bos2 is Conj[bos]:
        if bos in ('E','Estar'):
            return complex(E[bos]-2*pi*b['l0'])
        elif bos in ('D','Dstar'):
            return complex(E[bos]-2*pi*sum(b[m] for m in ('l0','lup','ldn')))
        elif bos in ('up','dn'):
            return complex(E[bos]-2*pi*b['l0'] - 2*pi*b['l'+bos])
        elif bos in ('upstar','dnstar'):
            return complex((E[bos]-2*pi*b['l0'] - 2*pi*b['l'+Conj[bos]]))
    else:
        return 0
#Convenience maps
Conj = {'D':'Dstar','E':'Estar','up':'upstar','dn':'dnstar','Dstar':'D','Estar':'E','upstar':'up','dnstar':'dn'}
Not = {'up':'dn','dn':'up','upstar':'dnstar','dnstar':'upstar'}
bosons = ['E','D','up','dn','Estar','Dstar','upstar','dnstar']

N = 100
filling = int(N/2)
bwidth = 10
cstrength = 0.1
Eb = np.linspace(-bwidth,bwidth,N-1)
Eb = [complex(e) for e in Eb]
Vb = np.array([cstrength/sqrt(N-1)]*(N-1))     #Remember to normalize so that the coupling strength doesn't change when we add more bath sites
Vb = [complex(v) for v in Vb]

#Multiplet energies
delta = pi*sum([v**2 for v in Vb])
#print(delta)
#quit()
R_converged = []
Uval = []

'Initialize parameters for first run'
b = np.array([0+0*1j],dtype=[(amp,'complex64') for amp in ('D','E','up','dn','Dstar','Estar','upstar','dnstar','l0','lup','ldn')])
realvals = np.random.rand(4)
imvals = [10**-1]*4
#imvals = np.random.rand(4)
#imvals = [0]*4
B = [(r+i*1j)/(sqrt(r**2+i**2)*2) for r,i in zip(realvals,imvals)]      #All constraints are satisfied here
#B[-1] = -B[-2]                                                          #This is a requirement for the spin

#B = [(r+i*1j) for r,i in zip(realvals,imvals)]      #All constraints are satisfied here
B += [a.conj() for a in B]  + [0+0.1*1j] + [10**-4]*2
for i,v in enumerate(['D','E','up','dn','Dstar','Estar','upstar','dnstar','l0','lup','ldn']):
    b[v] = B[i]
for s in range(60):
    #donext = input("Next U? y/n ")
    '''
    if donext is 'n':
        break
    '''
    U=s*pi*delta*0.1
    #U=0.3
    Uval.append(U)
    #print("U = {}".format(U))
    mu = U/2
    E = np.zeros(1,dtype=[(bos,'complex64') for bos in bosons])
    E[0] = tuple([U-2*mu,0,-mu,-mu,U-2*mu,0,-mu,-mu])
    #Stores the SB amplitudes and lagrange multipliers

    results = np.copy(b)
    k=0
    cnt =True
    Rvals = []
    bosDens = [1]
    dens_up = []
    dens_dn = []
    #nextu = input("Next U? y/n ")
    nextu='y'
    if nextu is 'n':
        quit()
    cnt = 'y'
    iter = 0
    while cnt == 'y':
        'Set up Hamiltonian etc.'
        M = []
        R = []
        Rc = []
        dms = []
        for spin in('up','dn'):
            m = complex(1/np.sqrt((b[Conj['D']]*b['D'] + b[Conj[spin]]*b[spin])*(b[Conj['E']]*b['E'] + b[Conj[Not[spin]]]*b[Not[spin]]))).real
            M.append(m)
            r = complex(b[Conj['D']]*b[spin] + b['E']*b[Conj[Not[spin]]]).real
            rc = complex(b['D']*b[Conj[spin]] + b[Conj['E']]*b[Not[spin]]).real
            R.append(r)
            Rc.append(rc)
            #print(Vb[0]*m*r)
            #print(m,r,rc)
            #print(r*m*Vb[1])
            #print(r,rc)
            #H = np.diag([2*pi*complex(b['l'+spin])]+list(Eb))
            #print(b['lup'],b['ldn'])
            #H = np.diag([complex(b['l'+spin])]+list(Eb))
            H = np.diag([0] + list(Eb))
            H[0,1:] = [v*r*m for v in Vb]
            H[1:,0] = [v*rc*m for v in Vb]
            eVals,eVecs = sp.linalg.eigh(H)
            order = np.argsort(eVals)
            eVecs = eVecs[:,order]
            dms.append(np.matmul(eVecs[:,0:filling],np.conj(eVecs[:,0:filling]).transpose()))
        dens_up.append(dms[0][0,0])
        dens_dn.append(dms[1][0,0])
        Rvals.append(float(np.real(R[0]*Rc[0]*M[0]*M[0])))
        #print(dens_up[-1],Rvals[-1])
        #py.matshow(np.abs(H),cmap='hot')
        #py.show()
        C = [sum(d*k for d,k in zip(mat[0,1:],Vb)) for mat in dms]
        Cc = [sum(d*k for d,k in zip(mat[1:,0],Vb)) for mat in dms]
        a1 = ['up','dn','up','dn']
        a2 = [False,False,True,True]

        'Compute the function to be minimized. Should package this up into lambda functions'  
        term1 = lambda bos1 : sum(c*m*dR(b,spin,bos1,conj=x) for c,m,spin,x in zip(C+Cc,M+M,a1,a2))
        term2 = lambda bos1 : sum(c*r*dM(b,spin,bos1) for c,r,spin, in zip(C+Cc,R+Rc,a1))

        f0 = lambda b : 1-sum(complex(b[Conj[bos]]*b[bos]) for bos in ('E','D','up','dn'))
        f1 = lambda spin,mat,b: complex(mat[0,0]-b[Conj['D']]*b['D']-b[Conj[spin]]*b[spin])
        f2 = lambda bos,b: pref(b,bos) + term1(bos) + term2(bos)

        funcs = [f0] + [partial(f1,spin,mat) for mat,spin in zip(dms,('up','dn'))] + [partial(f2,bos) for bos in bosons]    #Note: this should be defined out of the loop
        'Compute the Jacobian'
        J = [0,0,0]+[complex(-b[Conj[bos]]) for bos in ('E','D','up','dn','Estar','Dstar','upstar','dnstar')]  
        for spin in ('up','dn'):
            j = [0,0,0]+[complex(-b[Conj[bos]]) if bos not in ('E','Estar',Not[spin],Conj[Not[spin]]) else 0 for bos in ('E','D','up','dn','Estar','Dstar','upstar','dnstar')]
            J=np.vstack([J,j])
        for bos in bosons:
            term3 = lambda bos1,bos2 : sum(c*m*d2r(spin,bos,bos2,conj=x)for c,m,spin,x in zip(C+Cc,M+M,a1,a2))  
            term4 = lambda bos1,bos2 : sum(c*r*d2m(b,spin,bos,bos2)for c,r,spin in zip(C+Cc,R+Rc,a1))
            term5 = lambda bos1,bos2 : sum(c*dR(b,spin,bos1,conj=x)*dM(b,spin,bos2) for c,spin,x in zip(C+Cc,a1,a2))
            expression = lambda bos1,bos2 : term3(bos1,bos2)+term4(bos1,bos2)+term5(bos1,bos2)+term5(bos2,bos1) 
            j = [-2*pi*b[Conj[bos]]] + [-2*pi*b[Conj[bos]] if bos not in ('E','Estar',Not[spin],Conj[Not[spin]]) else 0 for spin in ('up','dn')]    #Check second terms?
            j += [dPref(b,bos,bos2) + expression(bos,bos2) for bos2 in bosons]
            j = [complex(k) for k in j]
            J = np.vstack([J,j])
        J = sp.matrix(J)

        'Now perform the globally convergent Newton step'
        b,sizeofCorr = globalNewton(funcs,b,J, tol = 10**-4)
        bosDens.append(sum(b[Conj[bos]]*b[bos] for bos in ('E','D','up','dn')))
        iter+=1
        if iter%5==0 and sizeofCorr>10**-8:
            #print(sizeofCorr)
            cnt = 'y'
            #py.plot(Rvals,'x')
            #py.show()
            #cnt = input("Continue?, y/n")
        elif sizeofCorr<10**-8:
            cnt = 'n'
            #print(Rvals[-1])
            R_converged.append(sqrt(Rvals[-1]))
            print(R_converged[-1])
        #quit()
        #print("Particle density = {}".format(dms[0][0,0]))
        '''
        b_new = -inverse(J)*g
        mixing = 10**-3
        #mixing = 1
        for i,v in enumerate(('l0','lup','ldn','E','D','up','dn','Estar','Dstar','upstar','dnstar')):
            new = b[v] + complex(b_new[i])
            b[v] = (1-mixing)*b[v] + mixing*new
        '''
sp.savez('results_D={}_c={}_N={}'.format(bwidth,cstrength,N),Uvals=Uval,R=R_converged,delta=delta)
py.plot([u/(pi*delta) for u in Uval],R_converged,'x')
py.show()
        #results = np.vstack([results,b])
        #print(bosDens)