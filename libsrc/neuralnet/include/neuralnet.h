#pragma once

enum NeuralNetwork_Errors {
    SUCCESS,
    INVALID_ARGUMENT,
};

typedef struct {
    int neuronCount;
    int weightsPerNeuron;
    float* biases;
    float* weights;
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
    int *neuronsPerLayer;
} NeuralNetwork_CreateRequest;

typedef struct {
    int epochs;
    float learningRate;
    char* trainingDirectory;
    char* validationDirectory;
} NeuralNetwork_TrainRequest;

typedef struct {
    int inputCount;
    int outputBufferSize;
    float* inputs;
    float* output;
} NeuralNetwork_PropogateRequest;

typedef struct {
    char* filePath;
} NeuralNetwork_FileRequest;

typedef union {
    NeuralNetwork_CreateRequest* create;
    NeuralNetwork_PropogateRequest* propogate;
    NeuralNetwork_TrainRequest* train;
    NeuralNetwork_FileRequest* file;
} NeuralNetwork_Request;

NeuralNetwork_Error NeuralNetwork_getLastError();

void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request);

void NeuralNetwork_destroy(NeuralNetwork* network);

void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request);

void NeuralNetwork_propogate(NeuralNetwork* network, NeuralNetwork_PropogateRequest* request);

void NeuralNetwork_save(NeuralNetwork* network, NeuralNetwork_FileRequest* request);

void NeuralNetwork_load(NeuralNetwork* network, NeuralNetwork_FileRequest* request);

void NeuralNetwork_print(NeuralNetwork* network);