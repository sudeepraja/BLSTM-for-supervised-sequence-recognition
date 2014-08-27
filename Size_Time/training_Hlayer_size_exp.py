from pylab import *
import numpy
import lstm
from lstm import *
import pickle
import os

lookup_table_file=open('lookup.txt','r')
symbols=[""," ","~"]
for line in lookup_table_file:
    if(line!='\n'):symbols.append(unichr(int(line[1:-1], 16)))
lookup_table_file.close()
c = Codec().init(symbols)

exp=open('Hlayer_size_exp.txt','w')
sizes=[10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000,1500,2000,3000]
results=[]
for siz in sizes:
    net=SeqRecognizer(15,siz,c.size(),c)
    
    #Output file to save the network
    output = open('network_new.pkl', 'wb')
    
    #List of Files containing training data, add more files seperated by ,
    files=['test2.txt']
    #No. of epochs
    epochs=2
    for y in range(epochs):
        for ff in files:
            f=open(ff, 'r')
            i=0;
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

    pickle.dump(net,output)
    v=float(os.path.getsize('network_new.pkl'))/1000000
    print net.Ni,net.Nh,net.No,v
    results.append(v)
    exp.write(str(net.Ni)+"  "+str(net.Nh)+"  "+str(net.No)+"  "+str(v)+"\n")
    output.close()

exp.close()
figure()
plot(sizes, results, 'g*-')
axis([0,3000+50 , 0, results[-1]+5])
xlabel('Nh')
ylabel('Mb')
title('No of hidden cells vs Size')
savefig('No of hidden cells vs Size.png')
