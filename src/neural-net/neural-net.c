#include "neural-net.h"
#include <memory.h>

void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request) {
    if (request->layerCount < 3) {
        lastError.type = INVALID_ARGUMENT;
        lastError.errorMessage = "Network must have 3 or more layers";
        return;
    }

    // Set the layer count
    network->layerCount = request->layerCount;
    
    // Initialize Each Layer
    for (int layer = 0; layer < request->layerCount; ++layer) {
        NeuronLayer* currentLayer = malloc(sizeof(*currentLayer));
        currentLayer->neuronCount = request->neuronsPerLayer[layer];

        network->layers[layer] = currentLayer;
    }
}

void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request) {

}

void NeuralNetwork_propogate(NeuralNetwork* network, NeuralNetwork_PropogateRequest* request) {

}

void NeuralNetwork_save(NeuralNetwork* network, NeuralNetwork_SaveRequest* request) {

}

void NeuralNetwork_load(NeuralNetwork* network, NeuralNetwork_LoadRequest* request) {

}