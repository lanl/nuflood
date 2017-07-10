#pragma once

#include <common/gpu_raster.hpp>

void UpdateFlux(GpuRaster<double>* topographic_elevation,
                GpuRaster<double>* water_surface_elevation);
