import json
from goodDesignHelpersAlt import *
import sys

if __name__ == '__main__':
    
    with open('../subSampleAltTran21.json','r') as file:
        subSampleCC = json.load(file)

    with open('../portraitString21.json','r') as file:
        portraitString21 = json.load(file)

    with open('../portraitIndexOfCC21.json','r') as file:
        portraitIndexOfCC21 = json.load(file)
        
    tempRet = []
    keys = list(subSampleCC.keys())
    lower = int(sys.argv[1])-1
    upper = lower+1
    #numberOfSamples = 2
    for v in [5,10,20]:
        portraitList = []
        v1 = v2 = v
        for k in keys[lower:upper]:
            index = portraitString21.index(k)
            heavisideIndex = portraitIndexOfCC21[index]
            psamples = subSampleCC[k]
            count=1
            for parameters in psamples:
                with open('log.dat','a') as file:
                    file.write(str(count)+'\n')
                count+=1
                # check the portrait of each sample for 
            #for key in keys:
                #parameters = parameterList[key]
                #print(k)
                theta1, theta2, mu, gamma, kappa, eta, e, f, = parameters
                H1= partial(Hill,v = v1)
                H2= partial(Hill,v = v2)
                numericalPF = partial(findFP,params = parameters,H1=H1,H2=H2)
                upperBound = (kappa/eta + gamma)*(1+eta)/(1+kappa)*2/mu*1/(e-f)
                samples = regionSample(theta1,theta2,upperBound)
                portrait = dict()
                #equilibrium = dict()
                for i in range(len(samples)):
                    for j in range(len(samples)):
                        p1, p2 = samples[i], samples[j]
                        x = fsolve(numericalPF,[p1,p2]) 

                        if not sum(map(lambda k: abs(k), numericalPF(x))) < 1e-6:
                            continue 
                        #print(x,numericalPF(x))
                        if isInRegion(x,parameters) and isAttractorOverall(x,parameters,H1,H2):
                            generatePortraits(x,parameters,portrait)

                        #if isInRegion(x,parameters):
                        #    generatePortraits(x,parameters,equilibrium)
                #print(portrait)
                portraitList.append(portrait)
        tempRet.append(portraitList)

    indexInf = []
    codeIndex = []

    for k in keys[lower:upper]:
        index = portraitString21.index(k)
        heavisideIndex = portraitIndexOfCC21[index]
        psamples = subSampleCC[k]
        for parameters in psamples:
            indexInf.append(heavisideIndex)
            codeIndex.append(k)
            #equilibriumList.append(equilibrium) 

    temp = []
    for t in tempRet:
        curr = [encode(x) for x in t]
        temp.append(curr)

    temp = [codeIndex] + temp + [indexInf]

    ret = []
    for i in range(len(psamples)):
        cur = [x[i] for x in temp]
        ret.append(cur)

    with open('result_{}.json'.format(lower+1),'w') as file:
        json.dump(ret,file)
        
       
    
