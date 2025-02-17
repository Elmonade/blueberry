CFLAGS = -Wall -Wextra -march=native -Ofast
CC = g++
VPATH = src:multiply/build

multiply/build/bin/matrix_mult : multiply/singleThread.o multiply/read.o
	mkdir -p multiply/build/bin
	$(CC) $(CFLAGS) $^ -o $@

multiply/build/bin/random_gen : multiply/randomNumberGenerator.o
	mkdir -p multiply/build/bin
	$(CC) $(CFLAGS) $^ -o $@

multiply/build/multiplier.o : multiply/multiplier.cpp
	mkdir -p multiply/build
	$(CC) $(CFLAGS) -c $? -o $@

multiply/build/singleThread.o : multiply/singleThread.cpp
	mkdir -p multiply/build
	$(CC) $(CFLAGS) -c $? -o $@

multiply/build/read.o : multiply/read.cpp
	$(CC) $(CFLAGS) -c $? -o $@

multiply/build/randomNumberGenerator.o : multiply/randomNumberGenerator.cpp
	$(CC) $(CFLAGS) -c $? -o $@

.PHONY = clean run generate

clean :
	rm -rvf multiply/build

run: multiply/build/bin/matrix_mult
	./multiply/build/bin/matrix_mult

generate: multiply/build/bin/random_gen
	./multiply/build/bin/random_gen

all: multiply/build/bin/matrix_mult multiply/build/bin/random_gen
