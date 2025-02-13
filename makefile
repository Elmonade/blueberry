CFLAGS = -Wall -Wextra
CC = g++
VPATH = src:build

build/bin/make : singleThread.o read.o
	mkdir build/bin
	$(CC) $(CFLAGS) $? -o $@

build/singleThread.o : singleThread.cpp
	mkdir build
	$(CC) $(CFLAGS) -c $? -o $@

build/read.o : read.cpp
	$(CC) $(CFLAGS) -c $? -o $@

.PHONY = clean
clean :
	rm -rvf build

.PHONY = run
run: make
	./build/bin/make