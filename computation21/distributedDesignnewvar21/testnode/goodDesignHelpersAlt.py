import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian
from numpy import heaviside
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
import json
from math import pow, exp
from scipy.optimize import fsolve, root
from functools import partial


# heavidside function
def Heaviside(x,theta):
    return 1 if x-theta >=0 else 0

# hill function
def Hill(x,theta,v):
    return 1/((theta/x)**v+ 1)


def isInRegion(p,params):
    p1 ,p2 = p
    theta1, theta2, mu, delta, eta1, eta2, e, f, = params
    
    first = (2/(mu*(e-f)) <= (p1+p2) <= 2*(1+eta1 + eta2)/(mu*(e-f)))
    second = (abs(p1-p2) <=2*(1+eta1 + eta2)/(mu*e))
    
    if first and second:
        return True
    else:
        False
        
        
def isAttractorOverall(p, parameters,H1,H2):
    theta1, theta2, mu, delta, eta1, eta2, e, f, = parameters
    
    def computeFixedPoint(p):
        p1,p2 = p
        a, b = 1 + delta - delta*H1(p1,theta1), 1+delta-delta*H1(p2,theta1)
        m1 = (mu+2*b)/(mu*(mu+a+b))
        m2 = 2/mu - m1
        
        return [m1,m2,p1,p2]
    
    fp = computeFixedPoint(p)
    
    def M1(fp):
        m1,m2,p1,p2 = fp
        a, b = 1 + delta - delta*H1(p1,theta1), 1+delta-delta*H1(p2,theta1)
        
        return 1-(mu+a)*m1 + b*m2

    def M2(fp):
        m1,m2,p1,p2 = fp
        a, b = 1 + delta - delta*H1(p1,theta1), 1+delta-delta*H1(p2,theta1)
        return 1 + a*m1 - (mu+b)*m2

    def P1(fp):
        m1,m2,p1,p2 = fp
        c,d = 1+ eta1*H1(p1,theta1)+eta2*H2(p1,theta2), 1+eta1*H1(p2,theta1)+eta2*H2(p2,theta2)
        return c*m1-e*p1+f*p2

    def P2(fp):
        m1,m2,p1,p2 = fp
        c,d = 1+ eta1*H1(p1,theta1)+eta2*H2(p1,theta2), 1+eta1*H1(p2,theta1)+eta2*H2(p2,theta2)
        return d*m2+f*p1-e*p2
    
    matrix = [grad(M1)(np.array(fp)),grad(M2)(np.array(fp)),grad(P1)(np.array(fp)),grad(P2)(np.array(fp))]
    vals = np.linalg.eigvals(matrix)
    
    for v in vals:
        if v>=0:
            return False
    return True

def regionSample(theta1,theta2,bound):
    lower = list(np.linspace(0,theta2,11)[1:])
    middle = list(np.linspace(theta2,theta1,11)[1:])
    bigger = list(np.linspace(theta1,max(bound,2*theta1),10))
    
    return lower+middle+bigger


# algebraic expression of fixed point for p1 and p2
# are we going to check the eigenvalues is negative or not
def findFP(p,params,H1,H2):
    
    p1 ,p2 = p
    theta1, theta2, mu, delta, eta1, eta2, e, f, = params
 
    a, b = 1 + delta - delta*H1(p1,theta1), 1+delta-delta*H1(p2,theta1)
    m1 = (mu+2*b)/(mu*(mu+a+b))
    m2 = 2/mu - m1
    
    c,d = 1+ eta1*H1(p1,theta1)+eta2*H2(p1,theta2), 1+eta1*H1(p2,theta1)+eta2*H2(p2,theta2)
    
    v1 = c*m1 - e*p1 + f*p2
    v2 = d*m2 + f*p1 - e*p2
        
    return v1, v2


# portraits is a dictionary of number fixed point in the corresponding 9 regions
def generatePortraits(p,parameters,portrait):
    theta1, theta2, mu, delta, eta1, eta2, e, f, =  parameters
    p1,p2 = p
    
    y, x = 0, 0
    
    if (p1 == theta1) or (p1==theta2) or (p2==theta1) or (p2==theta2):
        return 

    if p1 > theta2:
        if p1 < theta1:
            x = 1
        else:
            x = 2

    if p2 > theta2:
        if p2 < theta1:
            y = 1
        else:
            y = 2

    val = 3*y + x
    
    if val not in portrait:
        portrait[val] = set()
        
    roundP = tuple(map(lambda x: round(x,6),p))
    
    portrait[val].add(tuple(roundP))
    
def encode(portrait):
    ret = []
    for i in [3,6,7,0,4,8]:
        if i in portrait:
            ret.append(len(portrait[i]))
        else:
            ret.append(0)
    
    return ret