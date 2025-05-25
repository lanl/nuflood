#pragma once

#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif

#if USE_DOUBLE
using prec_t = double;
#else
using prec_t = float;
#endif

#ifdef BIG_GRID
#define INT_TYPE long unsigned int
#else
#define INT_TYPE unsigned int
#endif
