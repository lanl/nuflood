#pragma once

#include <common/isinks.h>
#include <common/timer.h>
#include "topography.h"
#include "time.h"
#include "output.h"
#include "conserved.h"
#include "constants.h"
#include "sources.h"
#include "infiltration.h"
#include "time_derivative.h"
#include "boundary_conditions.h"
#include "friction.h"
#include "slope.h"
#include "flux.h"
#include "active_cells.h"

class KurganovPetrova {
public:
	KurganovPetrova(const rapidjson::Value& root);
	void Step(void);
	void Run(void);
	void Print(void);

protected:
	Topography topography;
	Time time;
	Output output;
	Conserved conserved;
	Constants constants;
	ISinks sinks;
	Sources sources;
	Infiltration infiltration;
	TimeDerivative time_derivative;
	BoundaryConditions boundary_conditions;
	Friction friction;
	Slope slope;
	Flux flux;
	ActiveCells active_cells;
	Timer timer;
};
