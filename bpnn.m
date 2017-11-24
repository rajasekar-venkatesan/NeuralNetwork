function bpnn()
%MATLAB code for SLFN with BP
    
    
    %Network Initialization Function Begins
    function InitializeNetwork(inputnodes, hiddennodes, outputnodes, learningrate=0.9, activationfunction='sigmoid')
    %Network Initialization
    %Initialize a Neural Network with the given number of neurons in each layer
    %Usage:
    %   InitializeNetwork(inputnodes, hiddennodes, outputnodes, learningrate=0.9, activationfunction = 'sigmoid')
    %Arguements:
    %   inputnodes - Number of input layer neurons
    %   hiddennodes - Number of hidden layer neurons
    %   outputnodes - Number of output layer neurons
    %Optional Arguements:
    %   learningrate - Learning rate of the network (default: 0.9)
    %   activationfunction - Activation function of the network (default: sigmoid, currently supports: sigmoid)
    %Returns:
    %   Nil
    
        %Print status
        txt = fprintf('\nInitializing Neural network.\n');
        disp(txt)
        
        %Declare and assign global variables of the NN
        global inodes = inputnodes;
        global hnodes = hiddennodes;
        global onodes = outputnodes;
        global wih = randn(hnodes,inodes);
        global woh = randn(onodes,hnodes);
        global lr = learningrate;
        global actfn = activationfunction;
        
        %Print status
        txt = fprintf ('Neural network of shape (%d,%d,%d) is created.\n',inodes,hnodes,onodes);
        disp(txt)
    
    end
    %Network Initialization Function Ends
    

    %Activation Functions Definitions Begins
    function actfnout = sigmoid(inp,deriv=false)
    %Sigmoidal Activation Function
    %Usage:
    %   actfnout = sigmoid(inp,deriv=false)
    %Arguements:
    %   inp - Input to the activation function
    %   deriv - Boolean corresponding to actual activation function of derivative of the activation function
    %Returns:
    %   actfnout - Output of the activation function
    
        if not(deriv)
            actfnout = 1./(1+exp(-inp));
        else
            actfnout = sigmoid(inp);
            actfnout = actfnout.*(1-actfnout);
        end
        
    end
    %Activation Functions Definitions Ends


    %Calculate Layer Output in FwdPass Function Begins
    function layeroutputs = LayerOperation(layerinputs,deriv=false,layeractfn='sigmoid')
    %Layer Output Calculation - Applies activation function and calculate the output of neurons in corresponding layer
    %Usage:
    %   layeroutputs = LayerOperation(layerinputs,deriv=false,layeractfn='sigmoid')
    %Arguements:
    %   layerinputs - Input for the activation function
    %Optional Arguements:
    %   deriv - Boolean value corresponding to actual or derivative of activation function (default: false)
    %   layeractfn - Name of activation function (default: sigmoid)
    %Returns:
    %   layeroutputs - Output of neurons in the layer(aka output of activation function)
    
        switch layeractfn
            case 'sigmoid'
                layeroutputs = sigmoid(layerinputs,deriv);
            otherwise
                layeroutputs = sigmoid(layerinputs,deriv);
        end
        
    end
    %Calculate Layer Output in FwdPass Function Ends


    %Calculate FwdPass Function Begins
    function finaloutputs_trans = FwdPass(inputdata)
    %Forward Pass in the Network
    %Usage:
    %   finaloutputs_trans = FwdPass(inputdata)
    %Arguements:
    %   inputdata - Input samples in matrix form with each row corresponding to one sample and each column corresponding to one feature
    %Returns:
    %   finaloutputs_trans - Predicted output by the Network
    
        %Share global variables
        global inputs;
        global wih
        global woh;
        global actfn;
        global hiddeninputs;
        global hiddenoutputs;
        global finalinputs;
        global finaloutputs;
        
        %Forward pass calculation
        inputs = inputdata';

        hiddeninputs = wih*inputs;
        hiddenoutputs = LayerOperation(hiddeninputs);
        
        finalinputs = woh*hiddenoutputs;
        finaloutputs = LayerOperation(finalinputs);
        
        finaloutputs_trans = finaloutputs';
        
    end
    %Calculate FwdPass Function Ends


    function cost = BwdPass(outputactual)
        
        global finaloutputs;
        global finalinputs;
        global hiddeninputs;
        global hiddenoutputs;
        global wih;
        global woh;
        global lr;
        global inputs;
        global cost;
        
        outputerror = outputactual' - finaloutputs;
        outputactual_trans = outputactual';
        finaloutputs;
        outputerror;
        hiddenerror = woh'*outputerror;
        
        cost = sum(sum(outputerror.^2));
        
        outputdelta = outputerror.*LayerOperation(finalinputs,deriv=true);
        woh = woh + lr.*(outputdelta*hiddenoutputs');
        
        hiddendelta = hiddenerror.*LayerOperation(hiddeninputs,deriv=true);
        wih = wih + lr.*(hiddendelta*inputs');
        
    end
    
    function traincost = NetworkTrain(inputdata,outputactual,nepochs = 1000)
        for i = 1:nepochs
            FwdPass(inputdata);
            traincost = BwdPass(outputactual);
            
            if mod(i,10) == 0
                txt = sprintf('Cost at epoch %d is %0.5f\n', i,traincost);
                disp(txt)
            end
        end
    end
    
    function outputpred = NetworkPredict(inputdata)
        raw_outputpred = FwdPass(inputdata);
        outputpred = (raw_outputpred>0.5).*1;
    end
    


    %Create dataset - Binary input, XOR and AND output
    inputdata = [0 0;0 1;1 0;1 1];
    outputactual = [0 0 0 1;0 1 1 0;0 1 1 0;1 1 0 1];
    nepochs = 100;

    %Assign initialization values for the network to be created
    inputnodes = size(inputdata,2);
    hiddennodes = 6;
    outputnodes = size(outputactual,2);
    InitializeNetwork(inputnodes, hiddennodes, outputnodes);
    
    %Training and Prediction
    NetworkTrain(inputdata,outputactual,nepochs);
    %raw_outputpred = NetworkPredict(inputdata);
    %outputpred = (raw_outputpred>0.5).*1;
    outputpred = NetworkPredict(inputdata);
    
    inputdata
    outputactual
    outputpred
    
    txt = fprintf('Execution Completed.\n\n\n');
    disp(txt)
end
