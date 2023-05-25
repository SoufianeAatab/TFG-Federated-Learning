// This code is a modification of the code from http://robotics.hobbizine.com/arduinoann.html
#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK


/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int InputNodes = 650;
static const int HiddenNodes = 25;
static const int OutputNodes = 4;
static const float InitialWeightMax = 0.5;

typedef unsigned int uint;
static const uint hiddenWeightsAmt = (InputNodes + 1) * HiddenNodes;
static const uint outputWeightsAmt = (HiddenNodes + 1) * OutputNodes;

class NeuralNetwork {
    public:

        void initialize(float LearningRate, float Momentum);
        // ~NeuralNetwork();

        float forward(const float Input[], const float Target[]);
        float backward(const float Input[], const float Target[]);

        float* get_output();

        float* get_HiddenWeights();
        float* get_OutputWeights();

        float get_error();
        // float asd[500] = {};
        
    private:
        float Hidden[HiddenNodes] = {};
        float Output[OutputNodes] = {};
        float HiddenWeights[(InputNodes+1) * HiddenNodes] = {};
        float OutputWeights[(HiddenNodes+1) * OutputNodes] = {};
        float HiddenDelta[HiddenNodes] = {};
        float OutputDelta[OutputNodes] = {};
        float ChangeHiddenWeights[(InputNodes+1) * HiddenNodes] = {};
        float ChangeOutputWeights[(HiddenNodes+1) * OutputNodes] = {};

        float (*activation)(float);

        float Error;
        float LearningRate;
        float Momentum;
};


#endif

#include <math.h>

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}

void NeuralNetwork::initialize(float LearningRate, float Momentum) {
    this->LearningRate = LearningRate;
    this->Momentum = Momentum;

    this->activation = &sigmoid;
}

float NeuralNetwork::forward(const float Input[], const float Target[]){
    float error = 0;

    // Compute hidden layer activations
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = this->activation(Accum);
    }

    // Compute output layer activations and calculate errors
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[HiddenNodes*OutputNodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = this->activation(Accum);
        error += (1.0/OutputNodes) * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
    return error;
}

float NeuralNetwork::backward(const float Input[], const float Target[]){
    float error = 0;

    // Forward
    // Compute hidden layer activations
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = this->activation(Accum);
    }

    // Compute output layer activations and calculate errors
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = OutputWeights[HiddenNodes*OutputNodes + i];
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
        }
        Output[i] = this->activation(Accum);
        OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        error += (1.0/OutputNodes) * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
    // End forward

    // Backward
    // Backpropagate errors to hidden layer
    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        float Accum = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            Accum += OutputWeights[i*OutputNodes + j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
    }

    // Update Inner-->Hidden Weights
    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        HiddenWeights[InputNodes*HiddenNodes + i] += ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];
            HiddenWeights[j*HiddenNodes + i] += ChangeHiddenWeights[j*HiddenNodes + i] ;
        }
    }

    // Update Hidden-->Output Weights
    for(int i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes*OutputNodes + i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes*OutputNodes + i] ;
        OutputWeights[HiddenNodes*OutputNodes + i] += ChangeOutputWeights[HiddenNodes*OutputNodes + i] ;
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j*OutputNodes + i] ;
            OutputWeights[j*OutputNodes + i] += ChangeOutputWeights[j*OutputNodes + i] ;
        }
    }

    return error;
}


float* NeuralNetwork::get_output(){
    return Output;
}

float* NeuralNetwork::get_HiddenWeights(){
    return HiddenWeights;
}

float* NeuralNetwork::get_OutputWeights(){
    return OutputWeights;
}

float NeuralNetwork::get_error(){
    return Error;
}