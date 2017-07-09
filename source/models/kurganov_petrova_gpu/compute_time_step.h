#pragma once

#include <common/gpu_raster.hpp>

double ComputeTimeStep(GpuRaster<double>* max_speed, double desingularization);
