#include <stdbool.h>
#include <stdio.h>
#include "./neural-net/neural-net.h"

const int INPUT_BUFFER_SIZE = 1024;

void parseRequest(char* input, NeuralNetwork* network) {
    char* words[INPUT_BUFFER_SIZE];
    char intermediateBuffer[INPUT_BUFFER_SIZE];
}

int main(int argc, char* argv[]) {
    char inputBuffer[INPUT_BUFFER_SIZE];
    
    NeuralNetwork network;
    bool running = true;

    while (running) {
        fgets(inputBuffer, INPUT_BUFFER_SIZE, stdin);
        parseRequest(inputBuffer, &network);
    }
}