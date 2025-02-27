CXXFLAGS = -Wall -Wextra -march=native -Ofast -fopenmp -lopenblas
CC = g++
VPATH = src:multiply/build

multiply/build/bin/matrix_mult : multiply/singleThread.o multiply/read.o
	mkdir -p multiply/build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

multiply/build/bin/random_gen : multiply/randomNumberGenerator.o
	mkdir -p multiply/build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

multiply/build/bin/multi: multiply/multiThread.o multiply/read.o
	mkdir -p multiply/build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

multiply/build/bin/broadcasting: multiply/broadcasting.o multiply/read.o
	mkdir -p multiply/build/bin
	$(CC) $(CXXFLAGS) $^ -o $@

multiply/build/read.o : multiply/read.cpp
	$(CC) $(CXXFLAGS) -c $? -o $@

multiply/build/multiplier.o : multiply/multiplier.cpp
	mkdir -p multiply/build
	$(CC) $(CXXFLAGS) -c $? -o $@

multiply/build/singleThread.o : multiply/singleThread.cpp
	mkdir -p multiply/build
	$(CC) $(CXXFLAGS) -c $? -o $@

multiply/build/multiThread.o : multiply/multiThread.cpp
	mkdir -p multiply/build
	$(CC) $(CXXFLAGS) -c $? -o $@

multiply/build/broadcasting.o : multiply/broadcasting.cpp
	mkdir -p multiply/build
	$(CC) $(CXXFLAGS) -c $? -o $@


multiply/build/randomNumberGenerator.o : multiply/randomNumberGenerator.cpp
	$(CC) $(CXXFLAGS) -c $? -o $@

.PHONY = clean run generate multi

clean :
	rm -rvf multiply/build

run: multiply/build/bin/matrix_mult
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./multiply/build/bin/matrix_mult

generate: multiply/build/bin/random_gen
	./multiply/build/bin/random_gen

multi: multiply/build/bin/multi
	./multiply/build/bin/multi

broad: multiply/build/bin/broadcasting
	./multiply/build/bin/broadcasting

all: multiply/build/bin/matrix_mult multiply/build/bin/random_gen
