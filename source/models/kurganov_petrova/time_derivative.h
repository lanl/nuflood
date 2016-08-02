#pragma once

#include <common/document.h>
#include <common/file.h>
#include <common/grid.h>
#include <common/name_list.h>
#include <common/folder.h>
#include "topography.h"
#include "output.h"

class TimeDerivative {
public:
	TimeDerivative(const rapidjson::Value& root, const Topography& topography);
	void WriteGrids(const Output& output, const prec_t current_time);

	const Grid<prec_t>& w(void) const { return w_; }
	const Grid<prec_t>& hu(void) const { return hu_; }
	const Grid<prec_t>& hv(void) const { return hv_; }

private:
	File w_file_, hu_file_, hv_file_;
	prec_t w_value_, hu_value_, hv_value_;
	Grid<prec_t> w_, hu_, hv_;
};
