typedef int (*Function_ptr)(int, int, int);

Function_ptr func;

static __attribute__((noinline)) int B(int x, int y, int z) {
  return x+y+z;
}

static __attribute__((noinline)) int Indirect(int x, int y, int z) {
    return B(z, y, x) + y + z;
}

static __attribute__((noinline)) int A(int x, int y, int z){
  int retIndirect = func(x, y, z);
  return retIndirect + y + x;
}

__attribute__((noinline)) int run() {
  int s = 1;
  func = Indirect;
  for (int i=0; i<100000; ++i) {
    s += A(i, i+1000, i+56789);
  }
  return s;
}
