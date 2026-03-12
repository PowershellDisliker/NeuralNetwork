CC=gcc

SRC_DIR=./src/
BIN_DIR=./bin/

neuralnet: $(BIN_DIR)
	$(CC) $(SRC_DIR)* -c -g -o $(BIN_DIR)nn.o
	ar rcs $(BIN_DIR)lib$@.a $(BIN_DIR)*

$(BIN_DIR):
	mkdir $(BIN_DIR)

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)
