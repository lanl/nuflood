# Nuflood

## Build status

| [Linux][lin-link] | [Codecov][cov-link] |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/losalamos/nuflood.svg?branch=master "Travis build status"
[lin-link]: https://travis-ci.org/losalamos/nuflood "Travis build status"
[cov-badge]: https://codecov.io/gh/losalamos/nuflood/branch/master/graph/badge.svg
[cov-link]: https://codecov.io/gh/losalamos/nuflood

## Installation

Nuflood uses the following software as its dependencies:

* CMake >= 2.8
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

## Build and run the Docker image

```bash
docker build -t nuflood .
docker run -it --rm nuflood /bin/bash
```

## License
BSD-ish. See https://github.com/losalamos/nuflood/blob/master/LICENSE.md

LA-CC-15-008