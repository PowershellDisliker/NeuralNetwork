#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../include/neuralnet.h"

NeuralNetwork_Error lastError = {NN_SUCCESS, ""};

NeuralNetwork_Error NeuralNetwork_getLastError() {
    NeuralNetwork_Error errorCopy = {lastError.type, lastError.errorMessage};
    
    lastError.type = NN_SUCCESS;
    lastError.errorMessage = "";

    return errorCopy;
}

void applyActivationFunction(float* inputArray, float* outputArray, int N, enum NeuralNetwork_ActivationFunctions function) {
    if (function == RELU)    return NeuralNetwork_ReLU(inputArray, outputArray, N);
    if (function == SOFTMAX) return NeuralNetwork_SoftMax(inputArray, outputArray, N);
    if (function == SIGMOID) return NeuralNetwork_Sigmoid(inputArray, outputArray, N);
    if (function == LINEAR)  return NeuralNetwork_Linear(inputArray, outputArray, N);

    lastError.type = NN_INVALID_ARGUMENT;
    lastError.errorMessage = "Invalid Activation Function\n";
}

void getOutputLayerDelta(float* inputArray, float* outputArray, int N, enum NeuralNetwork_ActivationFunctions activationFunction) {
    if (activationFunction == RELU)    return NeuralNetwork_ReLUDerrivative(inputArray, outputArray, N);
    if (activationFunction == SOFTMAX) return NeuralNetwork_SoftMaxDerrivative(inputArray, outputArray, N);
    if (activationFunction == SIGMOID) return NeuralNetwork_SigmoidDerrivative(inputArray, outputArray, N);
    if (activationFunction == LINEAR)  return NeuralNetwork_LinearDerrivative(inputArray, outputArray, N);

    lastError.type = NN_INVALID_ARGUMENT;
    lastError.errorMessage = "Invalid Activation Function\n";
}

char* getActivationString(enum NeuralNetwork_ActivationFunctions activation) {
    if (activation == RELU)    return "ReLU";
    if (activation == SIGMOID) return "Sigmoid";
    if (activation == LINEAR)  return "Linear";
    if (activation == SOFTMAX) return "SoftMax";
    else return "Error";
}

int getLargestNeuronCount(NeuralNetwork* network) {
    int maxNeurons = -1;
    
    for (int layer = 1; layer < network->layerCount; ++layer) {
        if (maxNeurons < network->layers[layer]->neuronCount) maxNeurons = network->layers[layer]->neuronCount;
    }

    if (maxNeurons < 0) {
        lastError.errorMessage = "Could not find neuron count in any network layers.\n";
        lastError.type = NN_INVALID_ARGUMENT;
    }

    return maxNeurons;
}

float getMSE(float* outputs, float* expectedOutputs, int N) {
    float totalSquaredError = 0.0f;

    for (int i = 0; i < N; ++i) {
        float error = outputs[i] - expectedOutputs[i];
        totalSquaredError += error * error;
    }

    return totalSquaredError / N;
}

NeuralNetwork_Samples* getSamples(char* inputFilePath) {
    NeuralNetwork_Samples* toReturn = malloc(sizeof(*toReturn));

    FILE* inputFile = fopen(inputFilePath, "rb");

    if (!inputFile) {
        return NULL;
    }

    int sampleCount;

    fread(&sampleCount, sizeof(sampleCount), 1, inputFile);

    printf("Reading %d Samples\n", sampleCount);

    toReturn->samples = malloc(sizeof(*toReturn->samples) * sampleCount);

    for (int sampleNumber = 0; sampleNumber < sampleCount; ++sampleNumber) {
        NeuralNetwork_Sample* currentSample = malloc(sizeof(*toReturn->samples[sampleNumber]));
        
        fread(&currentSample->inputCount, sizeof(currentSample->inputCount), 1, inputFile);
        currentSample->inputs = malloc(sizeof(*currentSample->inputs) * currentSample->inputCount);
        fread(currentSample->inputs, sizeof(*currentSample->inputs), currentSample->inputCount, inputFile);
        fread(&currentSample->outputCount, sizeof(currentSample->outputCount), 1, inputFile);
        currentSample->outputs = malloc(sizeof(*currentSample->outputs) * currentSample->outputCount);
        fread(currentSample->outputs, sizeof(*currentSample->outputs), currentSample->outputCount, inputFile);

        toReturn->samples[sampleNumber] = currentSample;
    }
    
    toReturn->sampleCount = sampleCount;
    return toReturn;
}

void NeuralNetwork_create(NeuralNetwork* network, NeuralNetwork_CreateRequest* request) {
    if (request->layerCount < 3) {
        lastError.type = NN_INVALID_ARGUMENT;
        lastError.errorMessage = "Network must have 3 or more layers";

        return;
    }

    // Set the layer count
    network->layerCount = request->layerCount;
    network->layers = malloc(sizeof(*network->layers) * request->layerCount);

    // Set the input layer size.
    network->layers[0] = malloc(sizeof(*network->layers[0]));
    network->layers[0]->neuronCount = request->neuronsPerLayer[0];
    network->activatedBuffers = malloc(sizeof(*network->activatedBuffers) * network->layerCount);
    network->unactivatedBuffers = malloc(sizeof(*network->unactivatedBuffers) * network->layerCount);
    
    // Initialize Each Layer
    for (int layer = 1; layer < request->layerCount; ++layer) {
        NeuronLayer* currentLayer = malloc(sizeof(*currentLayer));
        const int neuronCount = currentLayer->neuronCount;
    
        network->unactivatedBuffers[layer] = malloc(sizeof(*network->unactivatedBuffers[layer]) * neuronCount);
        network->activatedBuffers[layer] = malloc(sizeof(*network->activatedBuffers[layer]) * neuronCount);
        
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
        
        network->layers[layer] = currentLayer;
    }
}

void NeuralNetwork_destroy(NeuralNetwork* network) {
    free(network->layers[0]);

    for (int layer = 0; layer < network->layerCount; ++layer) {
        free(network->unactivatedBuffers[layer]);
        free(network->activatedBuffers[layer]);

        network->unactivatedBuffers[layer] = NULL;
        network->activatedBuffers[layer] = NULL;
    }
    
    for (int layer = 1; layer < network->layerCount; ++layer) {
        free(network->layers[layer]->weights);
        free(network->layers[layer]->biases);
        free(network->layers[layer]);
    
        network->layers[layer]->weights = NULL;
        network->layers[layer]->biases = NULL;
        network->layers[layer] = NULL;
    }
    
    free(network->unactivatedBuffers);
    free(network->activatedBuffers);
    free(network->layers);
    
    network->layers = NULL;
    network->activatedBuffers = NULL;
    network->layers = NULL;

    network->layerCount = -1;
}

void NeuralNetwork_train(NeuralNetwork* network, NeuralNetwork_TrainRequest* request) {    
    // Create network output buffer
    int bufferSize = getLargestNeuronCount(network);
    int outputBufferSize = network->layers[network->layerCount - 1]->neuronCount;
    float* outputBuffer = malloc(sizeof(*outputBuffer) * outputBufferSize);
    
    NeuralNetwork_PropagateRequest innerRequest;
    innerRequest.outputBufferSize = outputBufferSize;
    innerRequest.output = outputBuffer;
    
    // Create delta buffers
    float* nextLayerError = malloc(sizeof(*nextLayerError) * bufferSize);
    float* currentLayerError = malloc(sizeof(*currentLayerError) * bufferSize);

    // Create weight and bias update buffers
    float* weightBuffers[network->layerCount - 1];
    float* biasBuffers[network->layerCount - 1];

    // Initialize update buffers
    for (int layer = 1; layer < network->layerCount; ++layer) {
        NeuronLayer* currentLayer = network->layers[layer];
        int totalLayerWeights = currentLayer->neuronCount * currentLayer->weightsPerNeuron;
        int totalLayerBiases = currentLayer->neuronCount;

        if (totalLayerWeights < 0 || totalLayerBiases < 0) {
            lastError.errorMessage = "Overflow while calculating total weights or biases in layer\n";
            lastError.type = NN_OVERFLOW_ERROR;

            return;
        }

        weightBuffers[layer] = malloc(sizeof(*weightBuffers[layer]) * totalLayerWeights);
        biasBuffers[layer] = malloc(sizeof(*biasBuffers[layer]) * totalLayerBiases);
        
        if (weightBuffers[layer] == NULL || biasBuffers[layer] == NULL) {
            lastError.errorMessage = "Error allocating space for weight or bias update buffer\n";
            lastError.type = NN_ALLOCATION_ERROR;
        
            return;
        }

        memset(weightBuffers[layer], 0.0f, totalLayerWeights);
        memset(biasBuffers[layer], 0.0f, totalLayerBiases);
    }

    // Get Training Data
    NeuralNetwork_Samples* samples = getSamples(request->trainingFilePath);
    
    // training loop
    for (int epoch = 0; epoch < request->epochs; ++epoch) {
        for (int sample = 0; sample < samples->sampleCount; ++sample) {
            NeuralNetwork_Sample* currentSample = samples->samples[sample];

            // Mini-batch complete (Expensive?)
            if ((epoch * samples->sampleCount + sample + 1) % request->samplesPerWeightUpdate == 0) {

                // update network weights and biases
                for (int layer = network->layerCount - 1; layer > 0; ++layer) {
                    NeuronLayer* currentLayer = network->layers[layer];

                    // Update network weights
                    for (int neuron = 0; neuron < currentLayer->neuronCount; ++neuron) {
                        for (int weight = 0; weight < currentLayer->weightsPerNeuron; ++weight) {
                            currentLayer->weights[neuron * currentLayer->weightsPerNeuron + weight] -= weightBuffers[layer][neuron];
                        }
                    }

                    // reset weights and biases buffer
                    memset(weightBuffers[layer], 0.0f, currentLayer->neuronCount * currentLayer->weightsPerNeuron);
                    memset(biasBuffers[layer], 0.0f, currentLayer->neuronCount);
                }
            }

            // backprop
            innerRequest.inputCount = currentSample->inputCount;
            innerRequest.inputs = currentSample->inputs;
            NeuralNetwork_propagate(network, &innerRequest);

            // Output Layer
            getOutputLayerDelta(network->unactivatedBuffers[network->layerCount - 1], currentLayerError, network->layers[network->layerCount - 1]->neuronCount, network->layers[network->layerCount - 1]->outputActivationFunction);

            // Get first delta vector
            for (int neuron = 0; neuron < network->layers[network->layerCount - 1]->neuronCount; ++neuron) {
                nextLayerError[neuron] += (innerRequest.output[neuron] - currentSample->outputs[neuron]) * network->unactivatedBuffers[network->layerCount - 1][neuron];
            }

            // Hidden Layers.
            for (int layer = network->layerCount - 2; layer >= 0; --layer) {
                // current and previous from a backwards perspective
                NeuronLayer* currentLayer = network->layers[layer];
                NeuronLayer* previousLayer = network->layers[layer + 1];

                applyActivationDerivative(network->unactivatedBuffers[layer], currentLayerError, currentLayer->neuronCount, currentLayer->outputActivationFunction);

                // Calculate current delta vector
                for (int currentNeuron = 0; currentNeuron < currentLayer->neuronCount; ++currentNeuron) {
                    float currentNeuronSum = 0.0f;
                    
                    for (int previousNeuron = 0; previousNeuron < previousLayer->neuronCount; ++previousNeuron) {
                        currentNeuronSum += previousLayer->weights[currentNeuron * currentLayer->neuronCount + previousNeuron] * nextLayerError[previousNeuron];
                    }

                    currentLayerError[currentNeuron] *= currentNeuronSum;
                }
                
                // update intermediate buffers
                for (int neuron = 0; neuron < currentLayer->neuronCount; ++neuron) {
                    biasBuffers[layer][neuron] += nextLayerError[neuron];

                    for (int weight = 0; weight < currentLayer->weightsPerNeuron; ++weight) {
                        weightBuffers[layer][neuron] += nextLayerError[neuron];
                    }
                }

                float* temp = nextLayerError;
                nextLayerError = currentLayerError;
                currentLayerError = temp;
            }
        }
    }

    // Final update if there is a remainder of samples * epoch / mini-batch size
    for (int layer = network->layerCount - 1; layer > 0; ++layer) {
        NeuronLayer* currentLayer = network->layers[layer];

        // Update network weights
        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            for (int weight = 0; weight < network->layers[layer]->weightsPerNeuron; ++weight) {
                currentLayer->weights[neuron * currentLayer->weightsPerNeuron + weight] -= weightBuffers[layer][neuron];
            }
        }
    }

    // Cleanup
    for (int layer = 1; layer < network->layerCount; ++layer) {
        free(weightBuffers[layer]);
        free(biasBuffers[layer]);
    }

    free(outputBuffer);
    free(previousLayerError);
    free(currentLayerError);
}

void NeuralNetwork_validate(NeuralNetwork *network, NeuralNetwork_ValidateRequest* request) {
    NeuralNetwork_Samples* samples = getSamples(request->validationFilePath);

    NeuralNetwork_PropagateRequest innerRequest;
    innerRequest.outputBufferSize = network->layers[network->layerCount - 1]->neuronCount;
    float outputBuffer[innerRequest.outputBufferSize];
    innerRequest.output = outputBuffer;
    float avgMSE = 0.0f;

    for (int sample = 0; sample < samples->sampleCount; ++sample) {
        NeuralNetwork_Sample* currentSample = samples->samples[sample];

        innerRequest.inputCount = currentSample->inputCount;
        innerRequest.inputs = currentSample->inputs;
        
        NeuralNetwork_propagate(network, &innerRequest);
    
        avgMSE += getMSE(outputBuffer, currentSample->outputs, innerRequest.outputBufferSize);

        free(currentSample->inputs);
        free(currentSample->outputs);
        free(currentSample);
    }

    request->mse = avgMSE / samples->sampleCount;

    free(samples->samples);
    free(samples);
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

    // load first intermediate buffers
    memcpy(network->activatedBuffers[0], request->inputs, sizeof(*network->activatedBuffers[0]) * network->layers[0]->neuronCount);

    // Propogate through the network
    for (int layer = 1; layer < network->layerCount; ++layer) {
        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            NeuronLayer* currentLayer = network->layers[layer];

            const int start = currentLayer->weightsPerNeuron * neuron;
            const int end = currentLayer->weightsPerNeuron * (neuron + 1);
            
            network->unactivatedBuffers[layer][neuron] = network->layers[layer]->biases[neuron];

            for (int innerProductIterator = start; innerProductIterator < end; ++innerProductIterator) {
                network->unactivatedBuffers[layer][neuron] += currentLayer->weights[innerProductIterator] * network->activatedBuffers[layer - 1][innerProductIterator - start];
            }

            applyActivationFunction(network->unactivatedBuffers[layer], network->activatedBuffers[layer], network->layers[layer]->neuronCount, network->layers[layer]->outputActivationFunction);
        }
    }

    // Write output to output buffer
    memcpy(request->output, network->activatedBuffers[network->layerCount - 1], sizeof(*request->output) * network->layers[network->layerCount - 1]->neuronCount);
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

    for (int layer = 0; layer < network->layerCount; ++layer) {
        network->activatedBuffers[layer] = malloc(sizeof(*network->activatedBuffers[layer]) * network->layers[layer]->neuronCount);
        network->unactivatedBuffers[layer] = malloc(sizeof(*network->unactivatedBuffers[layer]) * network->layers[layer]->neuronCount);
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

void NeuralNetwork_ReLU(float* input, float* output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = (input[i] < 0.0f) ? 0.0f : input[i];
    }
}

void NeuralNetwork_Linear(float* input, float* output, int N) {
    memcpy(output, input, N);
}

void NeuralNetwork_Sigmoid(float* input, float* output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

void NeuralNetwork_SoftMax(float* input, float* output, int N) {
    float sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        output[i] = expf(input[i]);
        sum += output[i];
    }

    for (int i = 0; i < N; ++i) {
        output[i] /= sum;
    }
}

void NeuralNetwork_ReLUDerrivative(float* input, float* output, int N) {
    for (int i = 0; i < N; ++i) {
        if (input[i] < 0) output[i] = 0;
        else output[i] = 1.0f;
    }
}

void NeuralNetwork_LinearDerrivative(float* input, float* output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = 1.0f;
    }
}

void NeuralNetwork_SigmoidDerrivative(float* input, float* output, int N) {
    for (int i = 0; i < N; ++i) {
        NeuralNetwork_Sigmoid(input, output, N);
    
        output[i] = output[i] * (1 - output[i]);
    }
}

void NeuralNetwork_SoftMaxDerrivative(float* input, float* output, int N) {
    float sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        output[i] = expf(input[i]);
        sum += output[i];
    }

    for (int i = 0; i < N; ++i) {
        output[i] /= sum;
    }
}