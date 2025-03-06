#CXXFLAGS = -Wall -Wextra -march=native -fopenmp -lopenblas
CXXFLAGS = -Wall -Wextra -march=native -Ofast -fopenmp -lopenblas
CC = g++
VPATH = src:multiply:build

build/bin/matrix_mult : build/singleThread.o build/read.o
	mkdir -p build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

build/bin/random_gen : build/randomNumberGenerator.o
	mkdir -p build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

build/bin/multi: build/multiThread.o build/read.o
	mkdir -p build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

build/bin/broadcasting: build/broadcasting.o build/read.o
	mkdir -p build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

build/read.o : multiply/read.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -c $? -o $@

build/multiplier.o : multiply/multiplier.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -c $? -o $@

build/singleThread.o : multiply/singleThread.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -c $? -o $@

build/multiThread.o : multiply/multiThread.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -c $? -o $@

build/broadcasting.o : multiply/broadcasting.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -c $? -o $@

build/randomNumberGenerator.o : multiply/randomNumberGenerator.cpp
	mkdir -p build
	$(CC) $(CXXFLAGS) -c $? -o $@

.PHONY = clean run generate multi broad all

clean :
	rm -rvf build

run: build/bin/matrix_mult
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build/bin/matrix_mult

generate: build/bin/random_gen
	./build/bin/random_gen

multi: build/bin/multi
	./build/bin/multi

broad: build/bin/broadcasting
	./build/bin/broadcasting

all: build/bin/matrix_mult build/bin/random_gen
