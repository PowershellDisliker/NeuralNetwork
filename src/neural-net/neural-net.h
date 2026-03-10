const enum NeuralNetwork_Errors {
    INVALID_ARGUMENT,
};


typedef struct {
    int weightCount;
    float bias;
    float weights[];
} Neuron;


typedef struct {
    int neuronCount;
    Neuron* neurons;
} NeuronLayer;


typedef struct {
    int layerCount;
    NeuronLayer* layers;
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

} NeuralNetwork_TrainRequest;


typedef struct {

} NeuralNetwork_PropogateRequest;


typedef struct {

} NeuralNetwork_SaveRequest;


typedef struct {

} NeuralNetwork_LoadRequest;


NeuralNetwork_Error lastError;


void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request);

void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request);

void NeuralNetwork_propogate(NeuralNetwork* network, NeuralNetwork_PropogateRequest* request);

void NeuralNetwork_save(NeuralNetwork* network, NeuralNetwork_SaveRequest* request);

void NeuralNetwork_load(NeuralNetwork* network, NeuralNetwork_LoadRequest* request);