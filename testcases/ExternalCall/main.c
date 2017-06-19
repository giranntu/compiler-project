#include <stdio.h>
#include <sys/time.h>
#include <dlfcn.h>

extern int safe_run();

double gt() {
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

extern int (*evil_ptr)();

int main() {
  int res;
  double start, end;

  void *self_handle = dlopen(NULL, 0);
  evil_ptr = (int (*)()) dlsym(self_handle, "evil_fn");

  start = gt();
  res = safe_run();
  end = gt();

  printf("res = %d\n", res);
  printf("time = %.3f\n", end - start);
}
