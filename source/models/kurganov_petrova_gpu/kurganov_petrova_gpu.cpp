#include <memory>
#include <common/document.h>
#include <common/error.h>
#include "kurganov_petrova_gpu.h"

KurganovPetrovaGpu::KurganovPetrovaGpu(const rapidjson::Value& root) { 
	std::string raster_path = root["rasterPath"].GetString();
	depth_ = new GpuRaster<float>(raster_path);
}

int main(int argc, char* argv[]) {
	// Check if a scenario file has been specified.
	if (argc <= 1) {
		PrintErrorAndExit("Scenario file has not been specified.");
	}

	//// Parse the scenario file as a JSON document.
	//Document json(scenario);

	//// Set up the model.

	// Parse the scenario file as a JSON document.
	Document json(argv[1]);

	// Set up the model.
	KurganovPetrovaGpu model(json.root["parameters"]);

	//// Run the model.
	//model.Run();

	//// Program executed successfully.
	//return 0;
}
