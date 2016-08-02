#pragma once

#include "file.h"
#include "grid.h"
#include "itime.h"
#include "name_list.h"
#include "timer.h"
#include "iconstants.h"
#include "isinks.h"
#include "itime.h"

class IConstants;
class ISinks;

class IOutput {
public:
	IOutput(const rapidjson::Value& root, const ITime& time);

	void IncrementTime(void);
	bool GridInList(const std::string grid_name) const;
	bool GridInList(const Grid<prec_t>& grid) const;
	void WriteGridIfInList(const prec_t current_time, const Grid<prec_t>& grid) const;
	void WriteGridIfInList(const Grid<prec_t>& grid) const;
	void PrintSummary(const ITime& time, const IConstants& constants, Timer& timer);
	void PrintInformation(const ITime& time);
	void WriteSinkTimeSeries(ISinks& sinks, const ITime& time, const IConstants& constants) const;

	NameList grid_list(void) const { return grid_list_; }
	Folder folder(void) const { return folder_; }
	prec_t time(void) const { return time_; }
	prec_t time_step(void) const { return time_step_; }
	bool print_summary(void) const { return print_summary_; }
	bool print_volume(void) const { return print_volume_added_ || print_volume_conservation_error_; }

protected:
	Folder folder_;
	NameList grid_list_;
	prec_t time_;
	prec_t time_step_;
	bool print_time_;
	bool print_time_step_;
	bool print_iteration_;
	bool print_volume_added_;
	bool print_volume_computed_;
	bool print_volume_conservation_error_;
	bool print_summary_;
	bool write_sink_time_series_;
};
