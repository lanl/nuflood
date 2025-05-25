# Nuflood

## Build status

| [Status][status-link]          | [Coverage][cov-link]      |
| :----------------------------: | :----------------------: |
| [![status-badge]][status-link] | [![cov-badge]][cov-link] |

[status-badge]: https://github.com/lanl/nuflood/workflows/CI/badge.svg
[status-link]: https://github.com/lanl/nuflood/actions?query=workflow%3ACI
[cov-badge]: https://codecov.io/gh/lanl/nuflood/branch/master/graph/badge.svg
[cov-link]: https://codecov.io/gh/lanl/nuflood

## Installation

Nuflood uses the following software as its dependencies:

* CMake >= 3.5
* gcc >= 4.8.4

To retrieve dependencies included as submodules (e.g., rapidjson), run:

```bash
git submodule update --init --recursive
```

Finally, compile the software:

```bash
mkdir build && cd build
cmake .. && make
```

To compile on Mac OSX, ensure non-Clang `gcc` and `g++` compilers are installed, then run
```bash
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/path/to/gcc -DCMAKE_CXX_COMPILER=/path/to/g++ ..
make
```

## Build and run the Docker image

```bash
docker build -t nuflood .
docker run -it --rm nuflood /bin/bash
```

## Usage at a glance
To simulate a modeling scenario, execute a command similar to
```bash
/path/to/nuflood/executables/kurganov_petrova example.json
```
where `/path/to/nuflood/executables/kurganov_petrova` is the system path to the compiled Kurganov-Petrova simulation model and `example.json` is a path to the scenario input file.

For scenario examples, please visit the [Nuflood examples project](https://github.com/tasseff/nuflood-examples).

## License
[BSD-ish](https://github.com/lanl/nuflood/blob/master/LICENSE.md)

LA-CC-15-008
