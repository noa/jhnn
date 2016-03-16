#! /usr/bin/env sh

th test/Linear.lua
th test/LookupTable.lua
th test/libjhu.lua

if [ -f "build/libcujhu.so" ]; then
    echo "\nRunning GPU tests..."
    th test/libcujhu.lua
else
    echo "\n[WARNING] libcujhu not built; skipping GPU tests"
fi


# eof
