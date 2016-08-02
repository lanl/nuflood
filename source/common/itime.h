#pragma once

#include "document.h"
#include "precision.h"

class ITime {
public:
	ITime(void);
	ITime(const rapidjson::Value& root);

	void Increment(void);
	prec_t start(void) const { return start_; }
	prec_t end(void) const { return end_; }
	prec_t current(void) const { return current_; }
	prec_t step(void) const { return step_; }
	long unsigned int max_iterations(void) const { return max_iterations_; }
	long unsigned int iteration(void) const { return iteration_; }
	void set_step(prec_t step) { step_ = step; }

protected:
	prec_t start_;
	prec_t end_;
	prec_t current_;
	prec_t step_;
	long unsigned int max_iterations_;
	long unsigned int iteration_;
};
