import sys
f=open(sys.argv[1],'r')
m=0
i=0
for line in f:
    if(i==3):
        feature = line[8:].split()
        timeSteps=int(feature[2])
        if(timeSteps>m):
            m=timeSteps
    i+=1
    i%=5
print m