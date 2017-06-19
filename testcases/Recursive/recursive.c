static __attribute__((noinline)) int D(int x, int y, int z) {
  return x+y+z;
}

static __attribute__((noinline)) int Recursive(int x, int y, int z) {
  if(x>0) {
      int ret = Recursive(x/10, y, z);
      return ret * z + y;
  }
  else {
      int ret = D(x, y, z);
      return ret + z *x + z;
  }
}

static __attribute__((noinline)) int C(int x, int y, int z) {
  return x+y+z;
}

static __attribute__((noinline)) int B(int x, int y, int z) {
  return C(x, y, z)+x+y+z;
}

static __attribute__((noinline)) int A(int x, int y, int z){
  int retRecursive =  Recursive(x, y, z);
  return retRecursive + y + z;
}

__attribute__((noinline)) int run() {
  int s = 1;
  for (int i=0; i<100000; ++i) {
    s += A(i, i+1, i+2);
  }
  return s;
}
