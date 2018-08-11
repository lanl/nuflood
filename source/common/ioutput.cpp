#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include "parameter.h"
#include "ioutput.h"

IOutput::IOutput(const rapidjson::Value& root, const ITime& time) {
	folder_.Clear();
	grid_list_.Clear();
	time_step_ = std::numeric_limits<prec_t>::max();

	print_time_ = false;
	print_time_step_ = false;
	print_iteration_ = false;

	print_volume_added_ = false;
	print_volume_computed_ = false;
	print_volume_conservation_error_ = false;

	print_summary_ = false;

	write_sink_time_series_ = false;

	if (root.HasMember("output")) {
		const rapidjson::Value& output_json = root["output"];
		ReadParameter(output_json, "folder", folder_);
		ReadParameter(output_json, "grids", grid_list_);
		ReadParameter(output_json, "timeStep", time_step_);
		ReadParameter(output_json, "printTime", print_time_);
		ReadParameter(output_json, "printTimeStep", print_time_step_);
		ReadParameter(output_json, "printIteration", print_iteration_);
		ReadParameter(output_json, "printVolumeComputed", print_volume_computed_);
		ReadParameter(output_json, "printVolumeAdded", print_volume_added_);
		ReadParameter(output_json, "printVolumeConservationError", print_volume_conservation_error_);
		ReadParameter(output_json, "printSummary", print_summary_);
		ReadParameter(output_json, "writeSinkTimeSeries", write_sink_time_series_);
	}

	if (time_step_ < time.end() - time.start()) {
		time_ = time.start();
	} else {
		time_ = std::numeric_limits<prec_t>::max();
	}
}

void IOutput::IncrementTime(void) {
	time_ += time_step_;
}


bool IOutput::GridInList(const std::string grid_name) const {
	return grid_list_.Contains(grid_name);
}

bool IOutput::GridInList(const Grid<prec_t>& grid) const {
	return grid_list_.Contains(grid.name());
}

void IOutput::WriteGridIfInList(const prec_t current_time, const Grid<prec_t>& grid) const {
	if (grid_list_.Contains(grid.name())) {
		grid.WriteWithoutBoundaries(folder_, current_time);
	}
}

void IOutput::WriteGridIfInList(const Grid<prec_t>& grid) const {
	if (grid_list_.Contains(grid.name())) {
		grid.WriteWithoutBoundaries(folder_);
	}
}

void IOutput::PrintSummary(const ITime& time, const IConstants& constants, Timer& timer) {
	timer.Stop();
	unsigned long num_cells = (unsigned long)constants.num_cells();
	double megacells_per_second = time.iteration()*num_cells / timer.SecondsElapsed() / 1e6;

	std::cout << std::left << std::setw(35);
	std::cout << "------------------- Summary -------------------" << std::endl;
	std::cout << std::setw(35) << "Number of iterations:"          << "\t";
	std::cout << std::setw(35) << time.iteration()                 << std::endl;
	std::cout << std::setw(35) << "Number of cells:"               << "\t";
	std::cout << std::setw(35) << num_cells                        << std::endl;
	std::cout << std::setw(35) << "Time elapsed (s):"              << "\t";
	std::cout << std::setw(35) << timer.SecondsElapsed()           << std::endl;
	std::cout << std::setw(35) << "Performance (megacells/s):"     << "\t";
	std::cout << std::setw(35) << megacells_per_second             << std::endl;
	std::cout << "-----------------------------------------------" << std::endl;
}


void IOutput::PrintInformation(const ITime& time) {
	static bool is_first_iteration = true;

	if (is_first_iteration) {
		if (print_time_) {
			std::cout << "Time (s)" << "\t";
		}

		if (print_time_step_) {
			std::cout << "Timestep (s)" << "\t";
		}

		if (print_iteration_) {
			std::cout << "Iteration" << "\t";
		}

		std::cout << std::endl;

		is_first_iteration = false;
	}

	if (print_time_) {
		std::cout << time.current() << "\t";
	}

	if (print_time_step_) {
		std::cout << time.step() << "\t";
	}

	if (print_iteration_) {
		std::cout << time.iteration() << "\t";
	}

	std::cout << std::endl;
}

void IOutput::WriteSinkTimeSeries(ISinks& sinks, const ITime& time, const IConstants& constants) const {
	std::ostringstream file_path;
	file_path << folder_.path() << "sinks.csv";
	std::ofstream output;

	static bool is_first_iteration = true;
	if (is_first_iteration) {
		output.open((file_path.str()).c_str());
		output.precision(std::numeric_limits<double>::digits10);

		output << "Time (s)" << ",";

		for (INT_TYPE k = 0; k < sinks.points().size(); k++) {
			output << sinks.points()[k].name();
			if (k < sinks.points().size() - 1) {
				output << ",";
			} else {
				output << std::endl;
			}
		}

		is_first_iteration = false;
		output.close();
	}

	output.open((file_path.str()).c_str(), std::ios_base::app);
	output.precision(std::numeric_limits<double>::digits10);
	output << time.current() << ",";
	for (INT_TYPE k = 0; k < sinks.points().size(); k++) {
		output << sinks.points()[k].depth() * constants.cellsize_x()*constants.cellsize_y();
		if (k < sinks.points().size() - 1) {
			output << ",";
		} else {
			output << std::endl;
		}
	}

	output.close();
}
