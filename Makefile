CC=gcc

EXEC_NAME=nni

LIB_DIR=./lib/
SRC_DIR=./src/
BIN_DIR=./bin/
LIB_SRC_DIR=./libsrc/

LIBS=neuralnet

LIB_PATHS = $(addprefix $(LIB_SRC_DIR)/, $(LIBS))
LDFLAGS = $(addprefix -l, $(LIBS))

all: $(LIBS) $(EXEC_NAME)

$(EXEC_NAME): $(SRC_DIR)cli.c
	-mkdir -p $(BIN_DIR)
	$(CC) $< -L$(LIB_DIR) $(LDFLAGS) -g -o $(BIN_DIR)$@ -lm

$(LIBS):
	$(MAKE) -C $(LIB_SRC_DIR)$@
	-mkdir $(LIB_DIR)
	cp $(LIB_SRC_DIR)$@/bin/lib$@.a $(LIB_DIR)

.PHONY: all clean $(LIBS)

clean:
	rm -rf $(BIN_DIR) $(LIB_DIR)
	@for dir in $(LIB_PATHS); do \
		$(MAKE) -C $$dir clean; \
	done