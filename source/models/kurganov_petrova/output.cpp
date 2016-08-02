#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <common/parameter.h>
#include "output.h"

Output::Output(const rapidjson::Value& root, const Time& time) : IOutput::IOutput(root, time) { }

void Output::PrintInformation(const Time& time, Conserved& U, const Topography& B, const Constants& C, const Infiltration& I) {
	static bool first_iteration = true;

	if (first_iteration) {
		if (print_time_) {
			std::cout << "Time (s)" << "\t";
		}

		if (print_time_step_) {
			std::cout << "Timestep (s)" << "\t";
		}

		if (print_iteration_) {
			std::cout << "Iteration" << "\t";
		}

		if (print_volume_added_) {
			std::cout << "Volume added (m^3)" << "\t";
		}

		if (print_volume_computed_) {
			std::cout << "Volume computed (m^3)" << "\t";
		}

		if (print_volume_conservation_error_) {
			std::cout << "Volume conservation error" << "\t";
		}

		std::cout << std::endl;
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

	if (print_volume_added_) {
		std::cout << U.volume_added() << "\t";
	}

	if (print_volume_computed_) {
		std::cout << U.volume_computed(C, I) << "\t";
	}

	if (print_volume_conservation_error_) {
		double error;

		if (U.volume_computed(C, I) == 0.0 && U.volume_added() == 0.0) {
			error = 0.0;
		} else {
			error = (U.volume_computed(C, I) - U.volume_added()) / (U.volume_added());
		}

		std::cout << error << "\t";
	}

	std::cout << std::endl;

	first_iteration = false;
}
