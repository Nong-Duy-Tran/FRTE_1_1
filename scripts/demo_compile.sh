#!/bin/bash
export LD_LIBRARY_PATH="${PWD}/../src/lib/opencv2":$LD_LIBRARY_PATH
cmake -S ../src/nullImpl/ -B ../build
cd ../build/
make
./main_exec