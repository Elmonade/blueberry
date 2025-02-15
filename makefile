CFLAGS = -Wall -Wextra -march=native
CC = g++
VPATH = src:matrixMult/build

matrixMult/build/bin/matrix_mult : matrixMult/singleThread.o matrixMult/read.o
	mkdir -p matrixMult/build/bin
	$(CC) $(CFLAGS) $^ -o $@

matrixMult/build/bin/random_gen : matrixMult/randomNumberGenerator.o
	mkdir -p matrixMult/build/bin
	$(CC) $(CFLAGS) $^ -o $@

matrixMult/build/singleThread.o : matrixMult/singleThread.cpp
	mkdir -p matrixMult/build
	$(CC) $(CFLAGS) -c $? -o $@

matrixMult/build/read.o : matrixMult/read.cpp
	$(CC) $(CFLAGS) -c $? -o $@

matrixMult/build/randomNumberGenerator.o : matrixMult/randomNumberGenerator.cpp
	$(CC) $(CFLAGS) -c $? -o $@

.PHONY = clean run generate

clean :
	rm -rvf matrixMult/build

run: matrixMult/build/bin/matrix_mult
	./matrixMult/build/bin/matrix_mult

generate: matrixMult/build/bin/random_gen
	./matrixMult/build/bin/random_gen

all: matrixMult/build/bin/matrix_mult matrixMult/build/bin/random_gen
