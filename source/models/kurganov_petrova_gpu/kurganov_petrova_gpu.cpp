#include <memory>
#include <common/document.h>
#include <common/error.h>
#include "kurganov_petrova_gpu.h"

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
}

KurganovPetrovaGpu::~KurganovPetrovaGpu(void) {
	delete topographic_elevation_;
	delete water_surface_elevation_;
	delete horizontal_discharge_;
	delete vertical_discharge_;
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

	// Program executed successfully.
	return 0;
}
