#include "kurganov_petrova.h"
#include "compute_flux.h"
#include "compute_timestep.h"
#include "integrate.h"
#include "update_boundaries.h"

KurganovPetrova::KurganovPetrova(const rapidjson::Value& root) :
  topography(root), time(root, topography.elevation_interpolated()),
  output(root, time), constants(root, topography),
  conserved(root, topography, constants, output), sinks(root, constants),
  sources(root, topography, constants),
  infiltration(root, topography), time_derivative(root, topography),
  boundary_conditions(root), friction(root),
  slope(topography.elevation_interpolated()),
  flux(topography.elevation_interpolated()),
  active_cells(conserved.w(), topography.elevation_interpolated(), sources, constants) {
}

void KurganovPetrova::Step(void) {
	sources.Update(time, constants, conserved.volume_added_);

	ComputeFlux(topography, sinks, sources, infiltration, constants, conserved,
	            slope, flux, time_derivative, time, active_cells);
	ComputeTimestep(active_cells, constants, time);

	// If the computed time step is greater than the temporal resolution at
	// which we wish to print data, set the step to the print interval, instead.
	if (time.step() > output.time_step()) {
		time.set_step(output.time_step());
	}

	Integrate(topography, time_derivative, friction, constants, time, conserved, infiltration, active_cells);
	UpdateBoundaries(boundary_conditions, constants, topography, time, conserved, sources);
	active_cells.Update(conserved.w(), topography.elevation_interpolated());
	time.Increment();
	Print();
}

void KurganovPetrova::Run(void) {
	timer.Reset();

	// Print initial data.
	conserved.WriteGrids(output, time.current());
	output.PrintInformation(time, conserved, topography, constants, infiltration);
	output.IncrementTime();

	while (time.current() < time.end() &&
	       time.iteration() < time.max_iterations()) {
		Step();
	}

	// Write updated maximal data before exiting.
	conserved.WriteMaxGrids(output);

	// Print summary statistics.
	output.PrintSummary(time, constants, timer);
}

void KurganovPetrova::Print(void) {
	if (time.current() >= output.time()) {
		// Write the user-specified output grids.
		conserved.WriteGrids(output, time.current());
		time_derivative.WriteGrids(output, time.current());
		flux.WriteGrids(output, time.current());
		topography.WriteGrids(output);

		// Output the sink data.
		if (output.write_sink_time_series()) {
			output.WriteSinkTimeSeries(sinks, time, constants);
		}

		// Print extra user-specified information.
		output.PrintInformation(time, conserved, topography, constants, infiltration);

		// Increment output.time_ by output.timestep_.
		output.IncrementTime();
	}
}

int main(int argc, char* argv[]) {
	// Check if a scenario file has been specified.
	if (argc <= 1) {
		PrintErrorAndExit("Scenario file has not been specified.");
	}

	// Read the scenario file.
	File scenario(argv[1]);

	// Parse the scenario file as a JSON document.
	Document json(scenario);

	// Set up the model.
	KurganovPetrova model(json.root["parameters"]);

	// Run the model.
	model.Run();

	// Program executed successfully.
	return 0;
}
