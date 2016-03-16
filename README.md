### Compilation ###

To install the C/CUDA functions:

```
$ luarocks make
```

This builds `libjhu` and `libcujhu`. If your machine does not have
CUDA installed, only the CPU version `libjhu` is built.

Then run tests:

```
$ ./run_all_tests.sh
```
