#pragma once

#include "active_cells.h"
#include "constants.h"
#include "time.h"

void ComputeTimestep(ActiveCells& A, const Constants& C, Time& T);
