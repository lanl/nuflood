#pragma once

#include "topography.h"
#include "time_derivative.h"
#include "friction.h"
#include "constants.h"
#include "time.h"
#include "conserved.h"
#include "infiltration.h"
#include "active_cells.h"

void Integrate(const Topography& B, const TimeDerivative& dUdt,
               const Friction& Sf, const Constants& C, const Time& T,
               Conserved& U, Infiltration& I, ActiveCells& A);
