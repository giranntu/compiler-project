int (*evil_ptr)(int, int, int);
const int N = 1000000;

__attribute__((noinline)) int A(int x, int y, int z) {
  return x * x + x * y + y * z;
}

int evil_fn(int x, int y, int z) {
  return A(x + 1, y + 2, z + 3) + x + y + z;
}

__attribute__((noinline)) int run() {
  int sum = 0;
  for (int i=0; i<N; i++) {
    sum += (*evil_ptr)(i, i+1, i+2) + i + 3;
  }
  return sum;
}
