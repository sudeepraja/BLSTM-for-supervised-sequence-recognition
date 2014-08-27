from pylab import *
import numpy
import lstm
from lstm import *
import pickle
import os
#First you need to create a codec of the language you are trying to recoginze.I have already created a codec for Telugu in lstm.py file

def code(i):
    n=zeros(50,dtype=int)
    n[i-1]=1
    return n
#Now initialize the Network 
#No. of inputs(First Arg) =300
#No. of hidden units in Hidden Layer(Second Arg) = 150
#No. of outputs(Third Arg) =Size of codec
#Fourth arg is the codec itself
exp=open('Inplayer_size_exp.txt','w')
sizes=[10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1500,2000,3000]
timesteps=10
results=[]
for siz in sizes:
    net=BIDILSTM(siz,100,50)
    net.setLearningRate(1e-1)
    inp=numpy.random.rand(timesteps,siz)
    out=code(random.randint(1,50))
    
    
    #Output file to save the network
    output = open('network_new.pkl', 'wb')
    
    net.train(inp,out)
    #Save the network for testing
    pickle.dump(net,output)
    v=float(os.path.getsize('network_new.pkl'))/1000000
    print siz,100,50,v
    results.append(v)
    exp.write(str(siz)+"  "+str(100)+"  "+str(50)+"  "+str(v)+"\n")
    output.close()
exp.close()
figure()
plot(sizes, results, 'g*-')
axis([0,3000+50 , 0, results[-1]+5])
xlabel('Ni')
ylabel('size')
title('No of input features vs Size')
savefig('No of input features vs Size.png')
