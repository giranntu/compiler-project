#include <stdio.h>
#include <sys/time.h>

extern int safe_run();

double gt() {
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
  int res;
  double start, end;

  start = gt();
  res = safe_run();
  end = gt();

  printf("res = %d\n", res);
  printf("time = %.3f\n", end - start);
}
