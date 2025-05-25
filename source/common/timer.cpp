#include "timer.h"

Timer::Timer()
#ifdef _WIN32
    : frequency_{}, start_{}, end_{}
#else
    : start_{}, end_{}
#endif
{
    Reset();
}

void Timer::Start() { Reset(); }

void Timer::Stop() {
#ifdef _WIN32
    QueryPerformanceCounter(&end_);
#else
    gettimeofday(&end_, nullptr);
#endif
}

void Timer::Reset() {
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

auto Timer::SecondsElapsed() const -> double {
#ifdef _WIN32
    return (end_.QuadPart - start_.QuadPart) / frequency_.QuadPart;
#else
    constexpr double MICROSECONDS_PER_SECOND = 1.0e6;
    long seconds = end_.tv_sec - start_.tv_sec;
    long useconds = end_.tv_usec - start_.tv_usec;
    return static_cast<double>(seconds) +
           (static_cast<double>(useconds) / MICROSECONDS_PER_SECOND);
#endif
}
