#!/bin/bash
if test -d "CppToPython"; then
    echo "CppToPython exists."
else
    git clone https://github.com/fvicini/CppToPython.git
fi
cd CppToPython
git submodule init
git submodule update
mkdir -p externals
cd externals
cmake -DINSTALL_VTK=OFF -DINSTALL_LAPACK=OFF ../gedim/3rd_party_libraries 
make -j4
cd ..
mkdir -p release
cd release 
path_1 = 
cmake -DBLAS_LIBRARIES= -DCMAKE_PREFIX_PATH="/root/TesinaMOR/CppToPython/externals/Main_Install/eigen3;/root/TesinaMOR/CppToPython/externals/Main_Install/triangle;/root/TesinaMOR/CppToPython/externals/Main_Install/tetgen;/root/TesinaMOR/CppToPython/externals/Main_Install/googletest" ../
make -j4 GeDiM4Py
cd ..