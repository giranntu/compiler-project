LLC = llc
PIN_ROOT ?= ../pin-3.0-76991-gcc-linux

all: indirect
	
indirect: main.c indirect.c safe_run.s
	clang indirect.c -o indirect.ll -S -emit-llvm
	opt -O3 indirect.ll -o indirect.1.ll
	$(LLC) -O3 indirect.1.ll -o indirect.s
	gcc main.c safe_run.s indirect.s -o indirect

run: $(TARGET)
	$(PIN_ROOT)/pin -t ./InstCount.so -- ./indirect

clean: 
	rm -rf indirect.ll indirect.1.ll indirect.s indirect
