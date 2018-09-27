## A fast simulator for surface water dynamics and flooding

## Build status

| [Linux][lin-link] | [Codecov][cov-link] |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/lanl/nuflood.svg?branch=master "Travis build status"
[lin-link]: https://travis-ci.org/lanl/nuflood "Travis build status"
[cov-badge]: https://codecov.io/gh/lanl/nuflood/branch/master/graph/badge.svg
[cov-link]: https://codecov.io/gh/lanl/nuflood

## Introduction

## Highlights in v0.0.1 (2018-09-26)

## Compatibility

## Installation

Nuflood uses the following software as its dependencies:

* [CMake](https://cmake.org/) as a general build tool.
* (optional) [CUDA](https://developer.nvidia.com/cuda-zone) for graphics processing unit parallelization.
* (optional) [Doxygen](http://www.doxygen.nl) to build documentation.

To compile the application:

1. Execute `git submodule update --init --recursive` to retrieve submodules.
2. Create a directory called `build` in the nuflood directory.
3. Change to the `build` directory and run `cmake ..` to configure the build. Windows users can do this with the cmake-gui application.
4. On Windows, build the solution in the build directory. On Linux, run `make` from the build directory.

## Usage at a glance

## License
[BSD-ish](https://github.com/lanl/nuflood/blob/master/LICENSE.md)

LA-CC-15-008
