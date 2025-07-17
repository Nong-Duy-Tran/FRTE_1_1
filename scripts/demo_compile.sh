#!/bin/bash

cmake -S ../src/nullImpl/ -B ../build
cd ../build/
make
./main_exec