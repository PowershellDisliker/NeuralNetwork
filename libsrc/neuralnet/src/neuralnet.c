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

void applyActivationFunction(float* outputArray, int outputSize, enum NeuralNetwork_ActivationFunctions function) {
    if (function == RELU)    return NeuralNetwork_ReLU(outputArray, outputSize);
    if (function == SOFTMAX) return NeuralNetwork_SoftMax(outputArray, outputSize);
    if (function == SIGMOID) return NeuralNetwork_Sigmoid(outputArray, outputSize);
    if (function == LINEAR)  return NeuralNetwork_Linear(outputArray, outputSize);

    lastError.type = NN_INVALID_ARGUMENT;
    lastError.errorMessage = "Invalid Activation Function\n";
}

void applyActivationDerivative(float* outputArray, int outputSize, enum NeuralNetwork_ActivationFunctions activationFunction) {
    if (activationFunction == RELU)    return NeuralNetwork_ReLUDerrivative(outputArray, outputSize);
    if (activationFunction == SOFTMAX) return NeuralNetwork_SoftMaxDerrivative(outputArray, outputSize);
    if (activationFunction == SIGMOID) return NeuralNetwork_SigmoidDerrivative(outputArray, outputSize);
    if (activationFunction == LINEAR)  return NeuralNetwork_LinearDerrivative(outputArray, outputSize);

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
    // Create weight update buffers
    float* weightBuffers[network->layerCount - 1];
    long* biasBuffers[network->layerCount - 1];
    int bufferSize = getLargestNeuronCount(network);
    
    float* outputBuffer = malloc(sizeof(*outputBuffer) * bufferSize);
    float* deltaBuffer = malloc(sizeof(*deltaBuffer) * bufferSize);
    float* derrivativeBuffer = malloc(sizeof(*deltaBuffer) * bufferSize);

    for (int layer = 1; layer < network->layerCount; ++layer) {
        NeuronLayer* currentLayer = network->layers[layer];

        weightBuffers[layer] = malloc(sizeof(*weightBuffers[layer]) * currentLayer->neuronCount * currentLayer->weightsPerNeuron);
        biasBuffers[layer] = malloc(sizeof(*biasBuffers[layer]) * currentLayer->neuronCount);
        
        if (weightBuffers[layer] == NULL || biasBuffers[layer] == NULL) {
            lastError.errorMessage = "Error allocating space for weight or bias update buffer\n";
            lastError.type = NN_ALLOCATION_ERROR;
        
            return;
        }

        memset(weightBuffers[layer], 0.0f, currentLayer->neuronCount * currentLayer->weightsPerNeuron);
        memset(biasBuffers[layer], 0.0f, currentLayer->neuronCount);
    }

    // Get Training Data
    NeuralNetwork_Samples* samples = getSamples(request->trainingFilePath);
    
    // training loop
    for (int epoch = 0; epoch < request->epochs; ++epoch) {
        for (int sample = 0; sample < samples->sampleCount; ++sample) {
            NeuralNetwork_Sample* currentSample = samples->samples[sample];

            // Mini-batch complete
            if ((sample + 1) % request->samplesPerWeightUpdate == 0) {

                // update network weights and biases
                for (int layer = network->layerCount - 1; layer > 0; ++layer) {
                    NeuronLayer* currentLayer = network->layers[layer];

                    // Update network weights
                    for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
                        for (int weight = 0; weight < network->layers[layer]->weightsPerNeuron; ++weight) {
                            currentLayer->weights[neuron] -= weightBuffers[layer][neuron];
                        }
                    }

                    // reset weights and biases buffer
                    memset(weightBuffers[layer], 0.0f, currentLayer->neuronCount * currentLayer->weightsPerNeuron);
                    memset(biasBuffers[layer], 0.0f, currentLayer->neuronCount);
                }
            }

            // backprop

            NeuralNetwork_PropagateRequest innerRequest;
            innerRequest.inputCount = currentSample->inputCount;
            innerRequest.inputs = currentSample->inputs;
            innerRequest.outputBufferSize = bufferSize;
            innerRequest.output = outputBuffer;

            NeuralNetwork_propagate(network, &innerRequest);

            memcpy(derrivativeBuffer, outputBuffer, bufferSize);

            applyActivationDerivative(derrivativeBuffer, network->layers[network->layerCount - 1]->neuronCount, network->layers[network->layerCount - 1]->outputActivationFunction);

            for (int neuron = 0; neuron < network->layers[network->layerCount - 1]->neuronCount; ++neuron) {
                deltaBuffer[neuron] += \
                (innerRequest.output[neuron] - currentSample->outputs[neuron]) * derrivativeBuffer[neuron];
            }

            // update buffers
            for (int layer = network->layerCount - 1; layer > 0; ++layer) {
                NeuronLayer* currentLayer = network->layers[layer];
                
                // update intermediate buffers
                for (int neuron = 0; neuron < currentLayer->neuronCount; ++neuron) {
                    biasBuffers[layer][neuron] += 0.0f;

                    for (int weight = 0; weight < currentLayer->weightsPerNeuron; ++weight) {
                        weightBuffers[layer][neuron] += request->learningRate * deltaBuffer[neuron];
                    }
                }
            }
        }
    }

    // Final update if there is a remainder of samples * epoch / mini-batch size
    for (int layer = network->layerCount - 1; layer > 0; ++layer) {
        NeuronLayer* currentLayer = network->layers[layer];

        // Update network weights
        for (int neuron = 0; neuron < network->layers[layer]->neuronCount; ++neuron) {
            for (int weight = 0; weight < network->layers[layer]->weightsPerNeuron; ++weight) {
                currentLayer->weights[neuron] -= weightBuffers[layer][neuron];
            }
        }
    }

    // Cleanup
    for (int layer = 1; layer < network->layerCount; ++layer) {
        free(weightBuffers[layer]);
        free(biasBuffers[layer]);
    }
    free(outputBuffer);
    free(deltaBuffer);
    free(derrivativeBuffer);
}

void NeuralNetwork_validate(NeuralNetwork *network, NeuralNetwork_ValidateRequest* request) {
    NeuralNetwork_Samples* samples = getSamples(request->validationFilePath);

    for (int sample = 0; sample < samples->sampleCount; ++sample) {
        NeuralNetwork_PropagateRequest innerRequest;
        NeuralNetwork_Sample* currentSample = samples->samples[sample];

        innerRequest.inputCount = currentSample->inputCount;
        innerRequest.inputs = currentSample->inputs;
        innerRequest.outputBufferSize = network->layers[network->layerCount - 1]->neuronCount;
        float outputBuffer[innerRequest.outputBufferSize];
        innerRequest.output = outputBuffer;
        
        NeuralNetwork_propagate(network, &innerRequest);
    
        request->mse = getMSE(outputBuffer, currentSample->outputs, innerRequest.outputBufferSize);

        free(currentSample->inputs);
        free(currentSample->outputs);
        free(currentSample);
    }

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

void NeuralNetwork_ReLUDerrivative(float* input, int N) {
    for (int i = 0; i < N; ++i) {
        if (input[i] < 0) input[i] = 0;
        else input[i] = 1.0f;
    }
}

void NeuralNetwork_LinearDerrivative(float* input, int N) {
    for (int i = 0; i < N; ++i) {
        input[i] = 1.0f;
    }
}

void NeuralNetwork_SigmoidDerrivative(float* input, int N) {
    for (int i = 0; i < N; ++i) {
        float inputCopy = input[i];
        NeuralNetwork_Sigmoid(&inputCopy, 1);
    
        input[i] = inputCopy * (1 - inputCopy);
    }
}

void NeuralNetwork_SoftMaxDerrivative(float* vector, int N) {
    float sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        vector[i] = expf(vector[i]);
        sum += vector[i];
    }

    for (int i = 0; i < N; ++i) {
        vector[i] /= sum;
    }
}