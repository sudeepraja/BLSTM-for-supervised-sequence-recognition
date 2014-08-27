from pylab import *
import numpy
import lstm
from lstm import *
from timeit import default_timer as timer

def code(i):
    n=zeros(50,dtype=int)
    n[i-1]=1
    return n

exp=open('Inplayer_time_exp.txt','w')
sizes=[10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1500,2000,3000]
timesteps=10
results=[]
for siz in sizes:
    net=BIDILSTM(siz,100,50)
    net.setLearningRate(1e-1)
    inp=numpy.random.rand(timesteps,siz)
    out=code(random.randint(1,50))

    start=timer()
    for i in range(300):
        net.train(inp,out)
    v=timer()-start
    print siz,100,50,v
    results.append(v)
    exp.write(str(siz)+"  "+str(100)+"  "+str(50)+"  "+str(v)+"\n")
exp.close()
figure()
plot(sizes, results, 'g*-')
axis([0,3000+50 , 0, results[-1]])
xlabel('Ni')
ylabel('Time')
title('No of input features vs Time')
savefig('No of input features vs Time.png')

