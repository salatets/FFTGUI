FFT using Cooley–Tukey algorithm for windows with GUI

This project configurated for build only in Visual Studio 2019

if you wanted to build project with different SDK, follow consider step:
* Project Properties -> C/C++ ->  Additional  Include  Directories  These must include SDK headers
* Project Properties → Linker →  Additional  Library  Directories   These must include SDK lib folder
* Project Properties → Linker → Input → Additional Dependencies     These must include OpenCL.lib

For test opencl in your machine build FFT project in exe

converter.py for convert *.cl file to c string literals

**TODO**
CMake project
WPF .NetCore
Qt frontend