#pragma once

#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif

#if USE_DOUBLE
typedef double prec_t;
#else
typedef float prec_t;
#endif

#ifdef BIG_GRID
#define INT_TYPE long unsigned int
#else
#define INT_TYPE unsigned int
#endif
