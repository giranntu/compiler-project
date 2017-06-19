LLC = llc

all: test1

test1: main.c test1.c safe_run.s
	clang test1.c -o test1.ll -S -emit-llvm
	opt -O3 test1.ll -o test1.1.ll -S
	$(LLC) -O3 test1.1.ll -o test1.s
	gcc main.c safe_run.s test1.s -o test1 -ldl -Wl,--export-dynamic

run: test1
	$(PIN_ROOT)/pin -t ./InstCount.so -- ./test1

clean:
	rm -rf test1.ll test1.1.ll test1.s test1
