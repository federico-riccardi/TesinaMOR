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
cd gedim/3rd_party_libraries
meshes.py
python cmake -DINSTALL_VTK=OFF -DINSTALL_LAPACK=OFF ../gedim/3rd_party_libraries 
python make -j4
cd ..




!cmake -DINSTALL_VTK=OFF -DINSTALL_LAPACK=OFF ../gedim/3rd_party_libraries
!make -j4
%cd ..
    "mkdir -p release",
    "cd release ",
    "cmake -DCMAKE_PREFIX_PATH= ../ ",
    "make -j4 GeDiM4Py",
    "cd .."