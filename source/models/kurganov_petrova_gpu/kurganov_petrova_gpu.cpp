#include <limits>
#include <memory>
#include <common/document.h>
#include <common/error.h>
#include "kurganov_petrova_gpu.h"
#include "compute_time_step.h"

KurganovPetrovaGpu::KurganovPetrovaGpu(const rapidjson::Value& simulation) {
	topographic_elevation_ = new GpuRaster<double>(simulation["topography"].GetString());
	water_surface_elevation_ = new GpuRaster<double>(*topographic_elevation_, "waterSurfaceElevation.tif");

	if (simulation.HasMember("horizontalDischarge")) {
		horizontal_discharge_ = new GpuRaster<double>(simulation["horizontalDischarge"].GetString());
	} else {
		horizontal_discharge_ = new GpuRaster<double>(*topographic_elevation_, "horizontalDischarge.tif");
		horizontal_discharge_->Fill(0.0);
	}

	if (simulation.HasMember("verticalDischarge")) {
		vertical_discharge_ = new GpuRaster<double>(simulation["verticalDischarge"].GetString());
	} else {
		vertical_discharge_ = new GpuRaster<double>(*topographic_elevation_, "verticalDischarge.tif");
		vertical_discharge_->Fill(0.0);
	}

	// Set time parameters.
	time_ = start_time_ = simulation["time"]["start"].GetDouble();
	end_time_ = simulation["time"]["end"].GetDouble();
	time_step_ = std::numeric_limits<double>::min();

	// Set constant parameters.
	if (simulation.HasMember("constants")) {
		if (simulation["constants"].HasMember("desingularization")) {
			desingularization_ = simulation["constants"]["desingularization"].GetDouble();
		} else {
			double dx = topographic_elevation_->get_cellsize_x();
			double dy = topographic_elevation_->get_cellsize_y();
			desingularization_ = sqrt(0.01*std::max(std::max(1.0, dx), dy));
		}

		if (simulation["constants"].HasMember("gravitationalAcceleration")) {
			gravitational_acceleration_ = simulation["constants"]["gravitationalAcceleration"].GetDouble();
		} else {
			gravitational_acceleration_ = 9.80665;
		}
	}
}

KurganovPetrovaGpu::~KurganovPetrovaGpu(void) {
	delete topographic_elevation_;
	delete water_surface_elevation_;
	delete horizontal_discharge_;
	delete vertical_discharge_;
}

void KurganovPetrovaGpu::Run(void) {
	while (time_ <= end_time_) {
		time_step_ = ComputeTimeStep(vertical_discharge_, desingularization_);
		time_ += time_step_;
	}	
}

int main(int argc, char* argv[]) {
	// Check if a scenario file has been specified.
	if (argc <= 1) {
		std::string error_message = "Path to JSON has not been specified.";
		std::cerr << "ERROR: " << error_message << std::endl;
		std::exit(1);
	}

	// Parse the scenario file as a JSON document.
	Document json(argv[1]);

	// Set up the model.
	rapidjson::Value& properties = json.root["properties"];
	KurganovPetrovaGpu simulation(properties["simulation"]);
	simulation.Run();

	// Program executed successfully.
	return 0;
}
