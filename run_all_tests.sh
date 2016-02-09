#! /usr/bin/env sh

if [ -f "build/libjhu.so" ]; then
    echo "\nRunning CPU tests..."
    th test/libjhu.lua
else
    echo "\n[WARNING] libjhu not built; skipping CPU tests"
fi

if [ -f "build/libcujhu.so" ]; then
    echo "\nRunning GPU tests..."
    th test/libcujhu.lua 
else
    echo "\n[WARNING] libcujhu not built; skipping GPU tests"
fi


# eof
