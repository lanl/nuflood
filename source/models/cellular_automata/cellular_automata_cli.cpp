#include <iostream>
#include "cellular_automata.h"

int main(int argc, char* argv[]) {
	// Check if a scenario file has been specified.
	if (argc > 1) {
		// Read in the input.
		Input input(argv[1]);

		// Set up the model.
		CellularAutomata cellular_automata(input);

		// Run the model.
		cellular_automata.Run();

		// Return code for successful execution.
		return 0;
	} else {
		std::string error_string = "Scenario file has not been specified.";
		std::cerr << error_string << std::endl;
		return 1; // Return code for undefined scenario.
	}
}
