#include <limits>
#include "itime.h"
#include "parameter.h"

ITime::ITime(const rapidjson::Value& root) {
	start_ = (prec_t)0;
	end_ = std::numeric_limits<prec_t>::max();
	step_ = std::numeric_limits<prec_t>::min();
	max_iterations_ = std::numeric_limits<long unsigned int>::max();

	if (root.HasMember("time")) {
		const rapidjson::Value& json = root["time"];
		ReadParameter(json, "start", start_);
		ReadParameter(json, "end", end_);
		ReadParameter(json, "step", step_);
		ReadParameter(json, "maxIterations", max_iterations_);
	}

	current_ = start_;
	iteration_ = 0;
}

void ITime::Increment(void) {
	current_ += step_;
	iteration_++;
}
