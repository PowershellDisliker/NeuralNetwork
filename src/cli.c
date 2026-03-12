#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "./neuralnet/include/neuralnet.h"

const int INPUT_BUFFER_SIZE = 1024;

void parseRequest(char* input, NeuralNetwork* network, bool* running) {
    char* words[INPUT_BUFFER_SIZE];
    char intermediateBuffer[INPUT_BUFFER_SIZE];

    int wordIterator = 0;
    int intermediateBufferIterator = 0;
    int inputIterator = 0;

    char currentChar = input[inputIterator++];
    
    while (currentChar != '\n' && currentChar != '\0') {
        if (currentChar == ' ') {
            intermediateBuffer[intermediateBufferIterator++] = '\0';
            words[wordIterator++] = malloc(sizeof(*words[wordIterator - 1]) * intermediateBufferIterator);
            intermediateBufferIterator = 0;
            continue;
        }

        intermediateBuffer[intermediateBufferIterator++] = currentChar;
        currentChar = input[inputIterator++];
    }

    for (int word = 0; word < wordIterator; ++word) {
        printf("%s", words[word]);
        free(words[word]);
    }
}

int main(int argc, char* argv[]) {
    char inputBuffer[INPUT_BUFFER_SIZE];
    
    NeuralNetwork network;
    bool running = true;

    while (running) {
        fgets(inputBuffer, INPUT_BUFFER_SIZE, stdin);
        parseRequest(inputBuffer, &network, &running);
    }
}