#include <stdbool.h>
#include "./neural-net/neural-net.h"

enum Operation {
    OPERATION_LOAD,
    OPERATION_TRAIN,
    OPERATION_SAVE,
    OPERATION_RUN,
    OPERATION_QUIT
};

typedef struct {
    // Common Options
    const enum Operation selectedOperation;
    
    int layerCount;
    int* neuronCounts;

    // Training Options
    char* trainingDataFilePath;
    char* validationDataFilePath;

    // Loading and Running Options
    float* inputs;

} CLI_Request;

void parseCLI_Request(CLI_Request *request, char* input) {

}

int main(int argc, char* argv[]) {
    CLI_Request requestBuffer;
    NeuralNetwork network;
    
    bool running = true;

    while (running) {
        char* input = fgets();
        parseCLI_Request(&requestBuffer, input);

        switch (requestBuffer.selectedOperation) {
            case OPERATION_LOAD:
            NeuralNetwork_load(&network);
            break;

            case OPERATION_TRAIN:
            NeuralNetwork_train(&network);
            break;

            case OPERATION_SAVE:
            NeuralNetwork_save(&network);
            break;

            case OPERATION_RUN:
            NeuralNetwork_propogate(&network);
            break;

            case OPERATION_QUIT:
            running = false;
            break;
        }
    }
}