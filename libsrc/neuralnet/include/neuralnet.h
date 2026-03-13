#pragma once

#include <stdbool.h>

enum NeuralNetwork_Errors {
    NN_SUCCESS,
    NN_INVALID_ARGUMENT,
    NN_IO_ERROR,
};

enum NeuralNetwork_ActivationFunctions {
    RELU,
    SOFTMAX,
    SIGMOID,
    LINEAR,
    NONE
};

typedef struct {
    int neuronCount;
    int weightsPerNeuron;
    float* biases;
    float* weights;
    enum NeuralNetwork_ActivationFunctions outputActivationFunction;
} NeuronLayer;

typedef struct {
    int layerCount;
    NeuronLayer** layers;
} NeuralNetwork;

typedef struct {
    enum NeuralNetwork_Errors type;
    char* errorMessage;
} NeuralNetwork_Error;

typedef struct {
    int layerCount;
    int* neuronsPerLayer;
    enum NeuralNetwork_ActivationFunctions* activationFunctions;
} NeuralNetwork_CreateRequest;

typedef struct {
    int epochs;
    float learningRate;
    char* trainingDirectory;
} NeuralNetwork_TrainRequest;

typedef struct {
    char* validationDirectory;
    float rmse;
} NeuralNetwork_ValidateRequest;

typedef struct {
    int inputCount;
    int outputBufferSize;
    float* inputs;
    float* output;
} NeuralNetwork_PropagateRequest;

typedef struct {
    char* filePath;
} NeuralNetwork_FileRequest;

typedef struct {
    int inputCount;
    float* inputs;
    int outputCount;
    float *outputs;
} NeuralNetwork_Sample;

typedef struct {
    int sampleCount;
    NeuralNetwork_Sample** samples;
} NeuralNetwork_Samples;

NeuralNetwork_Error NeuralNetwork_getLastError();

void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request);
void NeuralNetwork_destroy(NeuralNetwork* network);
void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request);
void NeuralNetwork_validate(NeuralNetwork* network, NeuralNetwork_ValidateRequest* request);
void NeuralNetwork_propagate(NeuralNetwork* network, NeuralNetwork_PropagateRequest* request);
void NeuralNetwork_save(NeuralNetwork* network, NeuralNetwork_FileRequest* request);
void NeuralNetwork_load(NeuralNetwork* network, NeuralNetwork_FileRequest* request);
void NeuralNetwork_print(NeuralNetwork* network);

void NeuralNetwork_ReLU(float* input, int N);
void NeuralNetwork_Sigmoid(float* input, int N);
void NeuralNetwork_Linear(float* input, int N);
void NeuralNetwork_SoftMax(float* vector, int N);