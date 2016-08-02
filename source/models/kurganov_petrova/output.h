#pragma once

#include <common/ioutput.h>
#include "time.h"
#include "conserved.h"
#include "topography.h"
#include "constants.h"
#include "infiltration.h"

class Conserved;
class Topography;
class Constants;
class Timer;
class Constants;
class Infiltration;

class Output : public IOutput {
public:
	Output(const rapidjson::Value& root, const Time& time);
	void PrintInformation(const Time& time, Conserved& U, const Topography& B, const Constants& C, const Infiltration& I);
};
