CFLAGS = -Wall -Wextra -march=native -O3 -fopenmp -lopenblas -mavx -mavx2 -mfma -fno-inline
CXXFLAGS += -I/usr/include/eigen3
CC = g++
VPATH = src:multiply/build

multiply/build/bin/matrix_mult : multiply/singleThread.o multiply/read.o
	mkdir -p multiply/build/bin
	$(CC) $(CFLAGS) $^ -o $@

multiply/build/bin/random_gen : multiply/randomNumberGenerator.o
	mkdir -p multiply/build/bin
	$(CC) $(CFLAGS) $^ -o $@

multiply/build/bin/multi: multiply/multiThread.o multiply/read.o
	mkdir -p multiply/build/bin
	$(CC) $(CFLAGS) $^ -o $@

multiply/build/read.o : multiply/read.cpp
	$(CC) $(CFLAGS) -c $? -o $@

multiply/build/multiplier.o : multiply/multiplier.cpp
	mkdir -p multiply/build
	$(CC) $(CFLAGS) -c $? -o $@

multiply/build/singleThread.o : multiply/singleThread.cpp
	mkdir -p multiply/build
	$(CC) $(CFLAGS) -c $? -o $@

multiply/build/multiThread.o : multiply/multiThread.cpp
	mkdir -p multiply/build
	$(CC) $(CFLAGS) -c $? -o $@


multiply/build/randomNumberGenerator.o : multiply/randomNumberGenerator.cpp
	$(CC) $(CFLAGS) -c $? -o $@

.PHONY = clean run generate multi

clean :
	rm -rvf multiply/build

run: multiply/build/bin/matrix_mult
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./multiply/build/bin/matrix_mult

generate: multiply/build/bin/random_gen
	./multiply/build/bin/random_gen

multi: multiply/build/bin/multi
	./multiply/build/bin/multi

all: multiply/build/bin/matrix_mult multiply/build/bin/random_gen
