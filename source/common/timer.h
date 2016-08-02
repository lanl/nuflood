#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class Timer {
public:
	Timer(void);
	void Start(void);
	void Stop(void);
	void Reset(void);
	double SecondsElapsed(void);

protected:
#ifdef _WIN32
	LARGE_INTEGER frequency_;
	LARGE_INTEGER start_;
	LARGE_INTEGER end_;
#else
	struct timeval start_;
	struct timeval end_;
#endif
};
