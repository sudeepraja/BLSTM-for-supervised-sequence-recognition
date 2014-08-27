from pylab import *
import numpy
import lstm
from lstm import *
import pickle
from timeit import default_timer as timer

lookup_table_file=open('lookup.txt','r')
symbols=[""," ","~"]
for line in lookup_table_file:
    if(line!='\n'):symbols.append(unichr(int(line[1:-1], 16)))
lookup_table_file.close()
c = Codec().init(symbols)

exp=open('Hlayer_time_exp.txt','w')
sizes=[10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1500,2000,3000]
results=[]
for siz in sizes:
    net=SeqRecognizer(15,siz,c.size(),c)
    
    ##OR you can also load a saved network for further training
    #n=open('network_old.pkl','r')
    #net=pickle.load(n)
    #n.close()
    
    
    #List of Files containing training data, add more files seperated by ,
    files=['test2.txt']
    #No. of epochs
    epochs=1
    start=timer()
    for y in range(epochs):
        for ff in files:
            f=open(ff, 'r')
            i=0
            for line in f:
                if(i==2):
                    truth = [unichr(int(k, 16)) for k in line[6:].split()]
                if(i==3):
                    feature = line[8:].split()
                    number=int(feature[0])
                    dimensions=int(feature[1])
                    timeSteps=int(feature[2])
                    features=array([float(j)for j in feature[3:]]).reshape(dimensions,timeSteps).transpose()
                if(i==4):
                    o=net.trainSequence(features,c.encode(truth))
                i+=1
                i%=5
            f.close()
    #Save the network for testing
    v=timer()-start
    print net.Ni,net.Nh,net.No,v
    results.append(v)
    exp.write(str(net.Ni)+"  "+str(net.Nh)+"  "+str(net.No)+"  "+str(v)+"\n")

exp.close()
figure()
plot(sizes, results, 'g*-')
axis([0,3000+50 , 0, results[-1]+5])
xlabel('Nh')
ylabel('time')
title('No of hidden cells vs Time')
savefig('No of hidden cells vs Time.png')
