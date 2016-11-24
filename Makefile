TF_DIR=$(HOME)/tensorflow
CONTRIB_DIR=$(TF_DIR)/tensorflow/contrib/makefile

TARGET=tf.o
SRC=tf.cpp
CC=gcc
CFLAGS=-std=c++11
INCLUDE_DIR=-I$(TF_DIR) -I$(CONTRIB_DIR)/gen/proto -I$(CONTRIB_DIR)/gen/protobuf/include -I$(CONTRIB_DIR)/downloads/eigen
INSTALL_DIR=$(HOME)/ffmpeg_build

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDE_DIR) -c $(SRC) -o $(TARGET)

install:
	cp $(TARGET) $(INSTALL_DIR)/lib/.
	cp tf.h $(INSTALL_DIR)/include/.

clean:
	rm $(TARGET)
