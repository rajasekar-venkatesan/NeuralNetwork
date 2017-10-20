'''Back Propagation Neural Network with Single Hidden Layer'''

#Imports
import numpy as np

#Defining Activation Functions and its Derivatives
def sigmoid(x,deriv = False):
    if not deriv:
        return 1 / (1+np.exp(-x))
    else:
        out = sigmoid(x)
        return out*(1-out)

#Neural Network Class
class BPNN:
    '''
    Class for Back propagation neural network with single hidden layer
    '''
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate = 0.9, activationfunction = sigmoid):
        '''
        Parameters
        ----------
        inputnodes : integer
            Number of neurons in input layer (equal to feature dimension) \n
        hiddennodes : integer
            Number of neurons in hidden layer \n
        outputnodes : integer
            Number of neurons in output layer (equal to number of outputs) \n
        learningrate : float
            Learning rate of the neural network (default: 0.9; currently learning rate is static) \n
        activationfunction : function
            Activation function for the neurons (default: sigmoid; currently only supports sigmoid) \n
        '''
        #Assigning neurons to layers
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #Initializing weight matrices
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes,self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes,self.hnodes))
        
        #Assigning learning rate and activation function
        self.lr = learningrate
        self.activationfunction = activationfunction

        #Print status
        print('\nNeural network of shape ({0}, {1}, {2}) is created.'.format(self.inodes,self.hnodes,self.onodes))
        
        pass
    
    def Fwd_Pass(self, input_data):
        '''
        Forward Pass Mechanism of the Neural Network.\n
        Data moves from input layer to output layer and prediction is made.\n
                
        Parameters
        ----------
        input_data : list
            2D List of input data with each row corresponding to one sample and each column corresponding to one feature
            
        Returns
        ---------
        output_pred : list
            Raw predicted output of the model with each row corresponding to one sample and each column corresponding to one output label
        
        '''
        #Convert the input list to array and transpose to faciliate matrix multiplication with weights
        self.inputs = np.array(input_data).T
        
        #Hidden layer input and outputs
        self.hidden_inputs = np.dot(self.wih, self.inputs)
        self.hidden_outputs = self.activationfunction(self.hidden_inputs)
        
        #Output layer input and outputs
        self.final_inputs = np.dot(self.who, self.hidden_outputs)
        self.final_outputs = self.activationfunction(self.final_inputs)
        
        #return predicted output
        return self.final_outputs.T
    
    def Bwd_Pass(self, output_actual):
        '''
        Backward Pass Mechanism of the Neural Network.\n
        Error moves from output layer to input layer and weights are updated. \n
                
        Parameters
        ----------
        output_actual : list
            List of actual output data with each row corresponding to one sample and each column corresponding to one feature
            
        Returns
        ---------
        cost : float
            Cost is calculated as sum of squared difference between actual output and predicted output for all samples
        
        '''
        #Calculate error values for each layer and cost
        output_error = np.array(output_actual).T - self.final_outputs
        self.cost = np.sum(output_error**2)
        hidden_error = np.dot(self.who.T, output_error)
        
        #Compute gradient and update weights for output layer 
        output_delta = output_error*self.activationfunction(self.final_inputs,deriv = True)
        self.who += self.lr * np.dot(output_delta,self.hidden_outputs.T)
        
        #Compute gradient and update weights for hidden layer
        hidden_delta = hidden_error*self.activationfunction(self.hidden_inputs,deriv = True)
        self.wih += self.lr * np.dot(hidden_delta,self.inputs.T)
        
        #return cost
        return self.cost
    
    #Public Functions
    def Train(self,input_data, output_actual, n_epochs = 10000):
        '''
        Train the Neural Network\n
        Performs forward pass based on input data for output prediction and backward pass based on output data for weight update and is repeated n_epochs number of times
        
        Parameters
        ----------
        input_data : list
            2D List of input data with each row corresponding to one sample and each column corresponding to one feature
        output_actual : list
            List of actual output data with each row corresponding to one sample and each column corresponding to one feature
        n_epochs : integer
            Number of iterations to train the network with input_data (default: 10000)
        
        Returns
        ---------
        cost : float
            Cost is calculated as sum of squared difference between actual output and predicted output for all samples
        
        '''
        #Print Status
        print('\nBegin Training...')
        
        #Do fwd and bwd pass n_epochs times
        for i in range(n_epochs+1):
            #Fwd and Bwd pass
            self.Fwd_Pass(input_data)
            cost = self.Bwd_Pass(output_actual)
            
            #Print loss periodically during training
            if (i % 1000) == 0:
                print('Cost at epoch {0} is {1:.6f}'.format(i,cost))
        
        #return cost    
        return cost
    
    def Predict(self, input_data):
        '''
        Predict the output using the Neural Network.\n
        For the given input data, make prediction using the Neural Network.\n
                
        Parameters
        ----------
        input_data : list
            2D List of input data with each row corresponding to one sample and each column corresponding to one feature
            
        Returns
        ---------
        output_pred : list
            Predicted output (0 or 1) of the model with each row corresponding to one sample and each column corresponding to one output label
        
        '''
        #Print Status
        print('\nBegin Prediction...')
        #Predict the output using Fwd_Pass for the input data
        output_pred = self.Fwd_Pass(input_data)
        
        #Convert the raw prediction output into 0 or 1 and return
        return (output_pred>0.5)*1
            


if __name__ == '__main__':
    
    #Create an object of BPNN class and initialize
    nnet = BPNN(2,6,2)
    
    #Create dataset (2 bit(inputs) XOR (output1) and AND(output2) operation)
    input_data = [[0,0],[0,1],[1,0],[1,1]]
    output_actual = [[0,0],[1,0],[1,0],[0,1]]
    
    #Train the Neural Network
    nnet.Train(input_data,output_actual)
    
    #Predict the output
    output_pred = nnet.Predict(input_data)
    
    #Print the prediction for each input
    for i in range(0,len(input_data)):
        print('Input: {0}, Predicted Output: {1}, Actual Output: {2}'.format(input_data[i],output_pred[i],output_actual[i]))
    
