#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../include/neuralnet.h"

NeuralNetwork_Error lastError = {NN_SUCCESS, ""};

NeuralNetwork_Error NeuralNetwork_getLastError() {
    NeuralNetwork_Error errorCopy = {lastError.type, lastError.errorMessage};
    
    lastError.type = NN_SUCCESS;
    lastError.errorMessage = "";

    return errorCopy;
}

void applyActivationFunction(float* outputArray, int outputSize, enum NeuralNetwork_ActivationFunctions function) {
    if (function == RELU)    return NeuralNetwork_ReLU(outputArray, outputSize);
    if (function == SOFTMAX) return NeuralNetwork_SoftMax(outputArray, outputSize);
    if (function == SIGMOID) return NeuralNetwork_Sigmoid(outputArray, outputSize);
    if (function == LINEAR)  return NeuralNetwork_Linear(outputArray, outputSize);

    lastError.type = NN_INVALID_ARGUMENT;
    lastError.errorMessage = "Invalid output activation function";
}

char* getActivationString(enum NeuralNetwork_ActivationFunctions activation) {
    if (activation == RELU)    return "ReLU";
    if (activation == SIGMOID) return "Sigmoid";
    if (activation == LINEAR)  return "Linear";
    if (activation == SOFTMAX) return "SoftMax";
    else return "Error";
}

void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request) {
    if (request->layerCount < 3) {
        lastError.type = NN_INVALID_ARGUMENT;
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
        currentLayer->outputActivationFunction = request->activationFunctions[layer];

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
    free(network->layers[0]);

    for (int layer = 1; layer < network->layerCount; ++layer) {
        free(network->layers[layer]->weights);
        free(network->layers[layer]->biases);
        free(network->layers[layer]);
    }

    free(network->layers);

    network->layerCount = -1;
    network->layers = NULL;
}

void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request) {
    
}

void NeuralNetwork_propagate(NeuralNetwork* network, NeuralNetwork_PropagateRequest* request) {
    // Validate Request
    if (network->layers[network->layerCount - 1]->neuronCount < request->outputBufferSize) {
        lastError.type = NN_INVALID_ARGUMENT;
        lastError.errorMessage = "Output Buffer too small for network";
        return;
    }

    if (network->layers[0]->neuronCount != request->inputCount) {
        lastError.type = NN_INVALID_ARGUMENT;
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
    for (int layer = 1; layer < network->layerCount; ++layer) {
        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            const int start = network->layers[layer]->weightsPerNeuron * neuron;
            const int end = network->layers[layer]->weightsPerNeuron * (neuron + 1);
            
            float innerProduct = network->layers[layer]->biases[neuron];

            for (int innerProductIterator = start; innerProductIterator < end; ++innerProductIterator) {
                innerProduct += network->layers[layer]->weights[innerProductIterator] * interInput[innerProductIterator - start];
            }

            interOutput[neuron] = innerProduct;
        }

        applyActivationFunction(interOutput, network->layers[layer]->neuronCount, network->layers[layer]->outputActivationFunction);

        float *temp = interInput;
        interInput = interOutput;
        interOutput = temp;
    }

    // Write output to output buffer
    for (int outputFeature = 0; outputFeature < network->layers[network->layerCount - 1]->neuronCount; ++outputFeature) {
        request->output[outputFeature] = intermediateInputBuffer[outputFeature];
    }
}

void NeuralNetwork_save(NeuralNetwork* network, NeuralNetwork_FileRequest* request) {
    FILE* outFile = fopen(request->filePath, "wb");

    fwrite(&network->layerCount, sizeof(network->layerCount), 1, outFile);

    fwrite(&network->layers[0]->neuronCount, sizeof(network->layers[0]->neuronCount), 1, outFile);

    for (int layer = 1; layer < network->layerCount; ++layer) {
        NeuronLayer* currentLayer = network->layers[layer];

        fwrite(&currentLayer->neuronCount, sizeof(currentLayer->neuronCount), 1, outFile);
        fwrite(&currentLayer->weightsPerNeuron, sizeof(currentLayer->weightsPerNeuron), 1, outFile);
        fwrite(&currentLayer->outputActivationFunction, sizeof(currentLayer->outputActivationFunction), 1, outFile);
        fwrite(currentLayer->weights, sizeof(*currentLayer->weights), currentLayer->neuronCount * currentLayer->weightsPerNeuron, outFile);
        fwrite(currentLayer->biases, sizeof(*currentLayer->biases), currentLayer->neuronCount, outFile);
    }

    fclose(outFile);
}

void NeuralNetwork_load(NeuralNetwork* network, NeuralNetwork_FileRequest* request) {
    FILE* inFile = fopen(request->filePath, "rb");

    if (inFile == NULL) {
        
        return;
    }

    fread(&network->layerCount, sizeof(network->layerCount), 1, inFile);
    network->layers = malloc(sizeof(*network->layers) * network->layerCount);
    network->layers[0] = malloc(sizeof(*network->layers[0]));
    fread(&network->layers[0]->neuronCount, sizeof(network->layers[0]->neuronCount), 1, inFile);


    for (int layer = 1; layer < network->layerCount; ++layer) {
        NeuronLayer* currentLayer = malloc(sizeof(*currentLayer));
        
        fread(&currentLayer->neuronCount, sizeof(currentLayer->neuronCount), 1, inFile);
        fread(&currentLayer->weightsPerNeuron, sizeof(currentLayer->weightsPerNeuron), 1, inFile);
        fread(&currentLayer->outputActivationFunction, sizeof(currentLayer->outputActivationFunction), 1, inFile);

        currentLayer->weights = malloc(sizeof(*currentLayer->weights) * currentLayer->neuronCount * currentLayer->weightsPerNeuron);
        currentLayer->biases = malloc(sizeof(*currentLayer->biases) * currentLayer->neuronCount);

        fread(currentLayer->weights, sizeof(*currentLayer->weights), currentLayer->neuronCount * currentLayer->weightsPerNeuron, inFile);
        fread(currentLayer->biases, sizeof(*currentLayer->biases), currentLayer->neuronCount, inFile);

        network->layers[layer] = currentLayer;
    }
}

void NeuralNetwork_print(NeuralNetwork* network) {
    if (network->layers == NULL) {
        printf("Network is empty\n");
        return;
    }

    printf("Input Layer:\n");
    printf("%d Input Neurons\n\n", network->layers[0]->neuronCount);

    for (int layer = 1; layer < network->layerCount; ++layer) {
        printf("Layer %d:\n", layer);
        printf("%d Neurons, %s Activation\n", network->layers[layer]->neuronCount, getActivationString(network->layers[layer]->outputActivationFunction));

        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            for (int weight = neuron * network->layers[layer]->weightsPerNeuron; weight < (neuron + 1) * network->layers[layer]->weightsPerNeuron; ++weight) {
                printf("%f ", network->layers[layer]->weights[weight]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void NeuralNetwork_ReLU(float* input, int N) {
    for (int i = 0; i < N; ++i) {
        input[i] = (input[i] < 0.0f) ? 0.0f : input[i];
    }
}

void NeuralNetwork_Linear(float* input, int N) {
    // Some of my best work.
    return;
}

void NeuralNetwork_Sigmoid(float* input, int N) {
    for (int i = 0; i < N; ++i) {
        input[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

void NeuralNetwork_SoftMax(float* vector, int N) {
    float sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        vector[i] = expf(vector[i]);
        sum += vector[i];
    }

    for (int i = 0; i < N; ++i) {
        vector[i] /= sum;
    }
}