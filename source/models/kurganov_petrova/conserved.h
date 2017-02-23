#pragma once

#include <common/document.h>
#include "constants.h"
#include "infiltration.h"
#include "output.h"
#include "topography.h"

class Output;
class Topography;
class Constants;
class Infiltration;

class Conserved {
public:
	Conserved(const rapidjson::Value& root, const Topography& topography,
	          const Constants& C, const Output& output);

	void WriteMaxGrids(const Output& output);
	void WriteGrids(const Output& output, const prec_t current_time);
	// TODO: Needs to be fixed to use dx and dy.
	double volume_computed(const Constants& C, const Infiltration& I);
	double volume_added(void) const { return volume_added_; }
	const Grid<prec_t>& w(void) const { return w_; }
	const Grid<prec_t>& w_old(void) const { return w_old_; }
	const Grid<prec_t>& hu(void) const { return hu_; }
	const Grid<prec_t>& hu_old(void) const { return hu_old_; }
	const Grid<prec_t>& hv(void) const { return hv_; }
	const Grid<prec_t>& hv_old(void) const { return hv_old_; }
	const Grid<prec_t>& h(void) const { return h_; }
	const Grid<prec_t>& h_max(void) const { return h_max_; }
	const Grid<prec_t>& q(void) const { return q_; }
	const Grid<prec_t>& q_max(void) const { return q_max_; }

	double volume_computed_, volume_added_;
	double sum_topography_heights_;

private:
	File w_file_, hu_file_, hv_file_, h_file_;
	prec_t w_value_, hu_value_, hv_value_, h_value_;
	Grid<prec_t> w_, w_old_, hu_, hu_old_, hv_, hv_old_;
	Grid<prec_t> h_, h_max_, q_, q_max_;
	Grid<prec_t> max_speed_;
};
