#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class Timer {
  public:
    Timer();
    void Start();
    void Stop();
    void Reset();
    auto SecondsElapsed() const -> double;

  private:
#ifdef _WIN32
    LARGE_INTEGER frequency_;
    LARGE_INTEGER start_;
    LARGE_INTEGER end_;
#else
    struct timeval start_;
    struct timeval end_;
#endif
};
