# FRTE 1:1

Please note that this package must be installed and run on Ubuntu 20.04.3 LTS, which can be downloaded from https://nigos.nist.gov/evaluations/ubuntu-20.04.3-live-server-amd64.iso.


# Null Implementation
There is a null implementation of the FRTE 1:1 API in ./src/nullImpl.  While the null implementation doesn't actually provide any real functionality, more importantly, it demonstrates mechanically how one could go about implementing, compiling, and building
a library against the API.

To compile and build the null implementation, from the top level validation directory run 

````console
$ ./scripts/build_null_impl.sh
````
This will place the implementation library into ./lib.
