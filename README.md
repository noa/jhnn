### Compilation ###

To install the C/CUDA functions:

```
#!bash

$ luarocks make
```

This builds `libjhu` and `libcujhu`. If your machine does not have
CUDA installed, only the CPU version `libjhu` is built.

Then run tests:

```
#!bash

$ ./run_all_tests.sh
```