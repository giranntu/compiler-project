LLC = llc
PIN_ROOT ?= ../pin-3.0-76991-gcc-linux

all: recursive
	
recursive: main.c recursive.c safe_run.s
	clang recursive.c -o recursive.ll -S -emit-llvm
	opt -O3 recursive.ll -o recursive.1.ll
	$(LLC) -O3 recursive.1.ll -o recursive.s
	gcc main.c safe_run.s recursive.s -o recursive

run: $(TARGET)
	$(PIN_ROOT)/pin -t ./InstCount.so -- ./recursive

clean: 
	rm -rf recursive.ll recursive.1.ll recursive.s recursive
