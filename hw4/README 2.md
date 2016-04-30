#### Building

Building this is basically the same as building CNN, you need to have Eigen

Create a symbolic link to cnn

    ln -s path/to/cnn/

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles 

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then to compile, run

    make -j 2
    ./morph_disambig ../train.txt ../test.txt 

