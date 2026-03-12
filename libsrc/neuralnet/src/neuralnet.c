#include "../include/neuralnet.h"
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>

NeuralNetwork_Error lastError;

void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request) {
    if (request->layerCount < 3) {
        lastError.type = INVALID_ARGUMENT;
        lastError.errorMessage = "Network must have 3 or more layers";
        return;
    }

    // Set the layer count
    network->layerCount = request->layerCount;
    NeuronLayer** layers = malloc(sizeof(*layers) * request->layerCount);

    // Set the input layer size.

    layers[0] = malloc(sizeof(*layers[0]));
    layers[0]->neuronCount = request->neuronsPerLayer[0];
    
    // Initialize Each Layer
    for (int layer = 1; layer < request->layerCount; ++layer) {
        NeuronLayer* currentLayer = malloc(sizeof(*currentLayer));

        currentLayer->neuronCount = request->neuronsPerLayer[layer];
        currentLayer->weightsPerNeuron = request->neuronsPerLayer[layer - 1];
        currentLayer->biases = malloc(sizeof(*currentLayer->biases) * currentLayer->neuronCount);
        currentLayer->weights = malloc(sizeof(*currentLayer->weights) * currentLayer->neuronCount * currentLayer->weightsPerNeuron);

        for (int i = 0; i < currentLayer->neuronCount; ++i) {
            currentLayer->biases[i] = (((float) rand() / (float) RAND_MAX) * 2) - 1.0f;
        }

        for (int i = 0; i < currentLayer->neuronCount * currentLayer->weightsPerNeuron; ++i) {
            currentLayer->weights[i] = (((float) rand() / (float) RAND_MAX) * 2) - 1.0f;
        }

        layers[layer] = currentLayer;
    }

    network->layers = layers;
}

void NeuralNetwork_destroy(NeuralNetwork* network) {
    for (int layer = 1; layer < network->layerCount; ++layer) {
        free(network->layers[layer]->weights);
        free(network->layers[layer]->biases);
        free(network->layers[layer]);
    }

    free(network->layers);
}

void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request) {
    
}

void NeuralNetwork_propogate(NeuralNetwork* network, NeuralNetwork_PropogateRequest* request) {
    // Validate Request
    if (network->layers[network->layerCount - 1]->neuronCount < request->outputBufferSize) {
        lastError.type = INVALID_ARGUMENT;
        lastError.errorMessage = "Output Buffer too small for network";
        return;
    }

    if (network->layers[0]->neuronCount != request->inputCount) {
        lastError.type = INVALID_ARGUMENT;
        lastError.errorMessage = "Input vector not the same size as network input";
        return;
    }

    // Get largest number of neurons in layer for intermediate output buffer
    int maxNeurons = -1;
    
    for (int layer = 0; layer < network->layerCount; ++layer) {
        if (network->layers[layer]->neuronCount > maxNeurons) maxNeurons = network->layers[layer]->neuronCount;
    }

    float intermediateInputBuffer[maxNeurons];
    float intermediateOutputBuffer[maxNeurons];

    float *interInput = intermediateInputBuffer;
    float *interOutput = intermediateOutputBuffer;

    // Preload the intermediate buffer with the provided input
    for (int inputFeature = 0; inputFeature < request->inputCount; ++inputFeature) {
        intermediateInputBuffer[inputFeature] = request->inputs[inputFeature];
    }

    // Propogate through the network
    for (int layer = 0; layer < network->layerCount; ++layer) {
        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            const int start = network->layers[layer]->weightsPerNeuron * neuron;
            const int end = network->layers[layer]->weightsPerNeuron * (neuron + 1);
            
            float innerProduct = network->layers[layer]->biases[neuron];

            for (int innerProductIterator = start; innerProductIterator < end; ++innerProductIterator) {
                innerProduct += network->layers[layer]->weights[innerProductIterator] * interInput[innerProductIterator - start];
            }

            interOutput[neuron] = innerProduct;
        }

        float *temp = interInput;
        interInput = interOutput;
        interOutput = temp;
    }

    // Write output to output buffer
    for (int outputFeature = 0; outputFeature < network->layers[network->layerCount - 1]->neuronCount; ++outputFeature) {
        request->output[outputFeature] = intermediateOutputBuffer[outputFeature];
    }
}

void NeuralNetwork_save(NeuralNetwork* network, NeuralNetwork_FileRequest* request) {

}

void NeuralNetwork_load(NeuralNetwork* network, NeuralNetwork_FileRequest* request) {

}

void NeuralNetwork_print(NeuralNetwork* network) {
    printf("Input Layer:\n");
    printf("%d Input Neurons\n\n", network->layers[0]->neuronCount);

    for (int layer = 1; layer < network->layerCount; ++layer) {
        printf("Layer %d:\n", layer);
        printf("%d Neurons\n", network->layers[layer]->neuronCount);

        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            for (int weight = neuron * network->layers[layer]->weightsPerNeuron; weight < (neuron + 1) * network->layers[layer]->weightsPerNeuron; ++weight) {
                printf("%f ", network->layers[layer]->weights[weight]);
            }
            printf("\n");
        }
        printf("\n");
    }
}