#include "timer.h"

Timer::Timer(void) {
	Reset();
}

void Timer::Start(void) {
	Reset();
}

void Timer::Stop(void) {
#ifdef _WIN32
	QueryPerformanceCounter(&end_);
#else
	gettimeofday(&end_, nullptr);
#endif
}

void Timer::Reset(void) {
#ifdef _WIN32
	start_.QuadPart = 0;
	end_.QuadPart = 0;
	QueryPerformanceFrequency(&frequency_);
	QueryPerformanceCounter(&start_);
	QueryPerformanceCounter(&end_);
#else
	gettimeofday(&start_, nullptr);
	gettimeofday(&end_, nullptr);
#endif
}

double Timer::SecondsElapsed(void) {
#ifdef _WIN32
	return (end_.QuadPart - start_.QuadPart) / frequency_.QuadPart;
#else
	long seconds  = end_.tv_sec  - start_.tv_sec;
	long useconds = end_.tv_usec - start_.tv_usec;
	return (double)seconds + (double)useconds / 1.0e6;
#endif
}
