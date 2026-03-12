#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../libsrc/neuralnet/include/neuralnet.h"

const int INPUT_BUFFER_SIZE = 1024;

void parseRequest_internal(char* input, NeuralNetwork* network, bool* running, char* words[], int wordCount) {
    char* command = words[0];

    if (strcmp(command, "create") == 0)
    {
        if (wordCount < 3)
        {
            printf("Create usage: create <layer_count> [input_n_count, l1_n_count, l2_n_count, ..., lk_n_count]\n\n");
            return;
        }

        int layers = atoi(words[1]);
        int neuronsPerLayer[layers];

        for (int layer = 0; layer < layers; ++layer)
        {
            neuronsPerLayer[layer] = atoi(words[2 + layer]);
        }

        NeuralNetwork_CreateRequest request;
        request.layerCount = layers;
        request.neuronsPerLayer = neuronsPerLayer;

        NeuralNetwork_create(network, &request);
        printf("Created %d-layer network\n", network->layerCount);
    }

    else if (strcmp(command, "run") == 0)
    {
        NeuralNetwork_PropogateRequest request;

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

        NeuralNetwork_propogate(network, &request);

        if (NeuralNetwork_getLastError().type != SUCCESS) {
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

    }
    
    else if (strcmp(command, "unload") == 0)
    {
        NeuralNetwork_destroy(network);

        printf("Freed Network\n");
    }

    else if (strcmp(command, "save") == 0)
    {

    }

    else if (strcmp(command, "load") == 0)
    {

    }

    else if (strcmp(command, "quit") == 0)
    {
        *running = false;
        printf("Goodbye!\n\n");
    }

    else if (strcmp(command, "print") == 0) {
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