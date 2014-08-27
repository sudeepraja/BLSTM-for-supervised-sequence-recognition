import numpy
from pylab import *
import lstm
from lstm import *
import pickle
import xml.etree.ElementTree as ET
import sys
from copy import deepcopy
from timeit import default_timer as timer

#create XML root node from first argument
tree = ET.parse(sys.argv[1])
root = tree.getroot()

#Function to avoid code bloating 
def TorF(s):
    if s=='True':return True
    else:return False

#get feature data from the list of files 
def get_data(_files):
    l=0
    for ff in _files:
        f=open(ff, 'r')
        for line in f:
            if(line!='\n'):
                l+=1
    data=[None]*(l//5)
    l=0
    for ff in _files:
        f=open(ff, 'r')
        i=0;
        #Read the file line by line
        for line in f:
            if(i==2):
                #Form the ground truth
                truth = [unichr(int(k, 16)) for k in line[6:].split()]
            if(i==3):
                #Create the feature vector
                feature = line[8:].split()
                number=int(feature[0])
                dimensions=int(feature[1])
                timeSteps=int(feature[2])
                features=array([float(j)for j in feature[3:]]).reshape(dimensions,timeSteps).transpose()
            if(i==4):
		#Save the feature
                data[l]=[number,truth,features]
                l+=1
            i+=1
            i%=5
    return data


#This function reads data from _files and predicts output or trains the network based on "PredOrTrain" string
#Additionally it outputs data into _output_file and _error_output_file depending on the booleans _output_file and _error_output
def Run(_data,_output,_output_file,_error_output,_error_output_file,PredOrTrain,epoch,net,verbose):
    #Initialize Errors
    label_errors=0
    word_errors=0
    words=0
    labels=0
    start = timer()
    if(_output):_output_file.write("Epoch :" + str(epoch+1)+"\n")

    for d in _data:
        number=d[0]
        truth = d[1]
        features=d[2]
        if(PredOrTrain=='Predict'):
            o=net.predictSequence(features)
        elif(PredOrTrain=='Train'):
            o=net.trainSequence(features,net.codec.encode(truth))
        inp=net.codec.encode(truth)
        out=[int(x) for x in o if x!=0.0]

        #Output the predictions to output file
        if(_output):
            _output_file.write("INPUT:"+str(number))
            _output_file.write("\n"+str(inp))
            if(verbose):_output_file.write("\n"+str([net.codec.code2char[x] for x in inp]))
            _output_file.write("\nOUTPUT:")
            _output_file.write("\n"+str(out))
            if(verbose):_output_file.write("\n"+str([net.codec.code2char[x] for x in out]))
            _output_file.write("\n\n")

        labels+=len(inp)
        words+=1
        #Find the error which is the levenshtein distance between inp and out
        errors=levenshtein(inp,out)
        if(errors>=1):
            label_errors+=errors
            word_errors+=1

    #Find the error rates for this epoch
    word_error_rate=float(word_errors)*100/words
    label_error_rate=float(label_errors)*100/labels

    time = timer() - start

    #Output the error rates to the output error file
    if(_error_output):
        _error_output_file.write("Epoch: "+str(epoch+1))
        _error_output_file.write("\nTotal words: "+str(words))
        _error_output_file.write("\nTotal word errors: "+str(word_errors))
        _error_output_file.write("\nWord error rate: "+str(word_error_rate))
        _error_output_file.write("\nTotal labels: "+str(labels))
        _error_output_file.write("\nTotal label errors: "+str(label_errors))
        _error_output_file.write("\nLabel error rate: "+str(label_error_rate))
        _error_output_file.write("\nTime for epoch: "+str(time))
        _error_output_file.write("\n\n")

    #Return the error rates for this epoch
    return word_error_rate,label_error_rate


def Training():
    new_or_old = root.findtext("NewOrOld")
    if(new_or_old=='New'):
        #Create a new network

        #Initialize Codec
        lookup_table_file=open(root.findtext("CodecPath"),'r')
        symbols=[""," ","~"]
        for line in lookup_table_file:
            if(line!='\n'):symbols.append(unichr(int(line[1:-1], 16)))
        lookup_table_file.close()
        codec = Codec().init(symbols)
        
        #Initialize Network
        directionality = root.findtext("Directionality")
        no_of_input_features = int(root.findtext("NoOfInputFeatures"))
        no_of_hidden_cells = [int(i) for i in root.findtext("NoOfHiddenCells").split(",")]
        net=SeqRecognizer(directionality,no_of_input_features,no_of_hidden_cells,codec.size(),codec)
        
        
    elif(new_or_old=='Old'):
        #Load the network specified in NetworkPath
        NetworkPath = root.findtext("NetworkPath")
        n=open(NetworkPath,'r')
        net=pickle.load(n)
        codec=net.codec
        n.close()
        
    #Set the Learning rate and momentum for training
    LearningRate = float(root.findtext("LearningRate"))
    Momentum=float(root.findtext("Momentum"))
    net.setLearningRate(LearningRate,Momentum)
    min_error_net=net
    min_lerror_rate=100.0
    min_werror_rate=100.0
    
    #Maximum number of epochs after which training is stopped
    max_epochs_for_training=int(root.findtext("MaximumEpochs"))
    
    #Check if validation data is available
    validation = TorF(root.findtext('Validation'))
    if(validation):
        #If Label error rate of validation data <= Threshold error rate, stop training
        ThresholdErrorRate = float(root.findtext('ThresholdErrorRate'))

        #Minimum number of epochs of training after validation occurs
        min_epochs_for_validation=int(root.findtext("MinimumEpochs"))
    
    #Collect the data for training and validating
    Training_data = get_data(root.findtext('TrainingFiles').split(','))
    if(validation):
        Validation_data = get_data(root.findtext('ValidationFiles').split(','))
    else:
        Validation_data = None

    #Check if output is verbose
    verbose= TorF(root.findtext('Verbose'))

    #Output file to record the training process
    Training_output = TorF(root.findtext('TrainingOutput'))
    Training_error_output = TorF(root.findtext('TrainingErrorOutput'))
    if(Training_output): Training_output_file = open(root.findtext('TrainingOutputPath'),'w')
    else:Training_output_file=None
    if(Training_error_output): Training_error_output_file = open(root.findtext('TrainingErrorOutputPath'),'w')
    else:Training_error_output_file=None
    
    #Output files to record the validation process
    if (validation):
        Validation_output = TorF(root.findtext('ValidationOutput'))
        Validation_error_output = TorF(root.findtext('ValidationErrorOutput'))
        if(Validation_output): Validation_output_file = open(root.findtext('ValidationOutputPath'),'w')
        else:Validation_output_file=None
        if(Validation_error_output): Validation_error_output_file = open(root.findtext('ValidationErrorOutputPath'),'w')
        else:Validation_error_output_file=None
    
    #Output file to save the network
    output = open(root.findtext('SaveNetworkPath'), 'wb')
    network_data = open(root.findtext('SaveNetworkDataPath'),'w')
    word_error_rate,label_error_rate=0,0
    
    #Train the network
    for epoch in range(max_epochs_for_training):
        word_error_rate,label_error_rate = Run(Training_data,Training_output,Training_output_file,Training_error_output,Training_error_output_file,'Train',epoch,net,verbose)
        if(label_error_rate<min_lerror_rate):
            min_lerror_rate=label_error_rate
            min_werror_rate=word_error_rate
            min_error_net = deepcopy(net)
        if(validation and epoch+1 >= min_epochs_for_validation):
            word_error_rate,label_error_rate = Run(Validation_data,Validation_output,Validation_output_file,Validation_error_output,Validation_error_output_file,'Predict',epoch,net,verbose)
            if(label_error_rate <= ThresholdErrorRate):
                break
            
    #Close the output file buffers
    if(Training_output):Training_output_file.close()
    if(Training_error_output):Training_error_output_file.close()
    if(validation and Validation_output):Validation_output_file.close()
    if(validation and Validation_error_output):Validation_error_output_file.close()

    #Save the network
    network_data.write("Network Information\n")
    network_data.write("No of input features: "+str(min_error_net.Ni)+"\n")
    network_data.write("No of hidden LSTM blocks in each side: "+str(min_error_net.Nh)+"\n")
    network_data.write("No of output classes: "+str(min_error_net.No)+"\n")
    network_data.write("Word Error Rate: "+str(min_werror_rate)+"\n")
    network_data.write("Label Error Rate: "+str(min_lerror_rate)+"\n")

    network_data.close()

    pickle.dump(min_error_net,output)
    output.close()

    #Save the network after last epoch
    SaveLastNetwork = TorF(root.findtext("SaveLastNetwork"))
    if(SaveLastNetwork):
        output = open(root.findtext('SaveLastNetworkPath'), 'wb')
        network_data = open(root.findtext('SaveLastNetworkDataPath'),'w')
        network_data.write("Network Information\n")
        network_data.write("No of input features: "+str(net.Ni)+"\n")
        network_data.write("No of hidden LSTM blocks in each side: "+str(net.Nh)+"\n")
        network_data.write("No of output classes: "+str(net.No)+"\n")
        network_data.write("Word Error Rate: "+str(word_error_rate)+"\n")
        network_data.write("Label Error Rate: "+str(label_error_rate)+"\n")

        network_data.close()

        pickle.dump(net,output)
        output.close()


def Testing():

    #Load the network specified in NetworkPath
    NetworkPath = root.findtext("NetworkPath")
    n=open(NetworkPath,'r')
    net=pickle.load(n)
    codec=net.codec
    n.close()

    #Collect the data for testing
    Testing_data = get_data(root.findtext('TestingFiles').split(','))

    #Output file to record the testing process
    Testing_output_file = open(root.findtext('TestingOutputPath'),'w')
    Testing_error_output_file = open(root.findtext('TestingErrorOutputPath'),'w')

    verbose= TorF(root.findtext('Verbose'))

    #Predict the outputs
    Run(Testing_data,True,Testing_output_file,True,Testing_error_output_file,'Predict',0,net,verbose)

    #Close the output file buffers
    Testing_output_file.close()
    Testing_error_output_file.close()

def main():
    Mode = root.findtext("Mode")
    if(Mode=='training'):
        Training()
    elif(Mode=='testing'):
        Testing()    
    
if __name__ == '__main__':
    main()
