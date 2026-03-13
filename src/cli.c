#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../libsrc/neuralnet/include/neuralnet.h"

const int INPUT_BUFFER_SIZE = 1024;

enum NeuralNetwork_ActivationFunctions getActivationEnumeration(char* word) {
    if (word == NULL) return NONE;

    if (strcasecmp(word, "relu") == 0)    return RELU;
    if (strcasecmp(word, "sigmoid") == 0) return SIGMOID;
    if (strcasecmp(word, "linear") == 0)  return LINEAR;
    if (strcasecmp(word, "softmax") == 0) return SOFTMAX;

    return NONE;
}

void parseRequest_internal(char* input, NeuralNetwork* network, bool* running, char* words[], int wordCount) {
    char* command = words[0];

    if (strcmp(command, "create") == 0)
    {
        if (wordCount < 4)
        {
            printf("Create usage: create <layer_count> [input_n_count l1_n_count l1_activation l2_n_count l2_activation ... lk_n_count lk_activation]\n\n");
            return;
        }

        int layers = atoi(words[1]);
        int neuronsPerLayer[layers];
        enum NeuralNetwork_ActivationFunctions activations[layers];

        neuronsPerLayer[0] = atoi(words[2]);
        activations[0] = NONE;

        for (int layer = 1; layer < layers; ++layer)
        {
            neuronsPerLayer[layer] = atoi(words[1 + (2 * layer)]);
            activations[layer] = getActivationEnumeration(words[2 + (2 * layer)]);
        }

        NeuralNetwork_CreateRequest request;
        request.layerCount = layers;
        request.neuronsPerLayer = neuronsPerLayer;
        request.activationFunctions = activations;

        NeuralNetwork_create(network, &request);
        printf("Created %d-layer network\n", network->layerCount);
    }

    else if (strcmp(command, "run") == 0)
    {
        NeuralNetwork_PropagateRequest request;

        request.inputCount = atoi(words[1]);
        request.outputBufferSize = network->layers[network->layerCount - 1]->neuronCount;
        
        float outputs[request.outputBufferSize];
        
        request.output = outputs;

        float inputs[request.inputCount];

        for (int i = 0; i < request.inputCount; ++i)
        {
            inputs[i] = atof(words[2 + i]);
        }

        request.inputs = inputs;

        NeuralNetwork_propagate(network, &request);

        if (NeuralNetwork_getLastError().type != NN_SUCCESS) {
            printf("Error calculating result");
            return;
        }

        printf("Results:\n");

        for (int result = 0; result < request.outputBufferSize; ++result) {
            printf("%f ", outputs[result]);
        }
        printf("\n");
     }
    
    else if (strcmp(command, "train") == 0)
    {
        if (wordCount == 1) {
            printf("Train Usage: train <epochs> <learning_rate> <input_data>\n");
            return;
        }

        NeuralNetwork_TrainRequest request;

        request.epochs = atoi(words[1]);
        request.learningRate = atof(words[2]);
        request.trainingDirectory = words[3];

        NeuralNetwork_train(network, &request);
    }
    
    else if (strcmp(command, "validate") == 0)
    {
        if (wordCount == 1) {
            printf("Validate Usage: validate <validation_data>\n");
            return;
        }

        NeuralNetwork_ValidateRequest request = {words[1], 0.0f};

        NeuralNetwork_validate(network, &request);

        printf("Root Mean Squared Error: %f\n", request.rmse);
    }

    else if (strcmp(command, "unload") == 0)
    {
        NeuralNetwork_destroy(network);

        printf("Freed Network\n");
    }

    else if (strcmp(command, "save") == 0)
    {
        NeuralNetwork_FileRequest request = {words[1]};
    
        NeuralNetwork_save(network, &request);
        printf("Network Saved.\n");
    }

    else if (strcmp(command, "load") == 0)
    {
        NeuralNetwork_FileRequest request = {words[1]};

        NeuralNetwork_load(network, &request);
        printf("Loaded Network.\n");
    }

    else if (strcmp(command, "quit") == 0)
    {
        *running = false;
        printf("Goodbye!\n\n");
    }

    else if (strcmp(command, "print") == 0)
    {
        NeuralNetwork_print(network);
    }
    
    else
    {
        printf("Command '%s' not recognized\n\n", command);
    }
}

void parseRequest(char* input, NeuralNetwork* network, bool* running)
{
    // Split the line of input into individual words
    char* words[INPUT_BUFFER_SIZE];
    char intermediateBuffer[INPUT_BUFFER_SIZE];

    int wordIterator = 0;
    int intermediateBufferIterator = 0;
    int inputIterator = 0;

    char currentChar = input[inputIterator++];
    
    while (currentChar != '\n' && currentChar != '\0')
    {
        if (currentChar == ' ')
        {
            intermediateBuffer[intermediateBufferIterator++] = '\0';
            words[wordIterator] = malloc(sizeof(*words[wordIterator]) * intermediateBufferIterator);

            for (int character = 0; character < intermediateBufferIterator; ++character)
            {
                words[wordIterator][character] = intermediateBuffer[character];
            }

            wordIterator++;
            intermediateBufferIterator = 0;
            currentChar = input[inputIterator++];
            continue;
        }

        intermediateBuffer[intermediateBufferIterator++] = currentChar;
        currentChar = input[inputIterator++];
    }

    if (intermediateBufferIterator > 0)
    {
        intermediateBuffer[intermediateBufferIterator++] = '\0';
        words[wordIterator] = malloc(sizeof(*words[wordIterator]) * intermediateBufferIterator);
        
        for (int character = 0; character < intermediateBufferIterator; ++character)
        {
            words[wordIterator][character] = intermediateBuffer[character];
        }

        wordIterator++;
    }

    // Start parsing the command
    parseRequest_internal(input, network, running, words, wordIterator);

    // cleanup
    for (int word = 0; word < wordIterator; ++word)
    {
        free(words[word]);
    }
}

int main(int argc, char* argv[])
{
    char inputBuffer[INPUT_BUFFER_SIZE];
    
    NeuralNetwork network;
    bool running = true;

    while (running)
    {
        fgets(inputBuffer, INPUT_BUFFER_SIZE, stdin);
        parseRequest(inputBuffer, &network, &running);
    }

    return 0;
}