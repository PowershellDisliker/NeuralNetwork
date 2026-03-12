CC=gcc

SRC_DIR=./src/
BIN_DIR=./bin/

nn: $(BIN_DIR)
	$(CC) $(SRC_DIR)* -c -o $(BIN_DIR)nn.o
	ar rcs ./nn $(BIN_DIR)*

$(BIN_DIR):
	mkdir $(BIN_DIR)

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)
