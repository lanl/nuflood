#pragma once

#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif

#if USE_DOUBLE
typedef double prec_t;
#else
typedef float prec_t;
#endif

#ifndef USE_LONG
#define USE_LONG 0
#endif

#if USE_LONG
typedef unsigned long int_t;
#else
typedef unsigned int int_t;
#endif
