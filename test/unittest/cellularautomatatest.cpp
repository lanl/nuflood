#include <models/cellular_automata/cellular_automata.h>
#include <models/cellular_automata/cellular_automata_input.h>
#include "gtest/gtest.h"

TEST(CellularAutomata, Constructor1) {
	CellularAutomataInput input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
}

TEST(CellularAutomata, Constructor2) {
	CellularAutomataInput input("../test/resources/cellular_automata_2.json");
	CellularAutomata cellular_automata(input);
}

TEST(CellularAutomata, Constructor3) {
	CellularAutomataInput input("../test/resources/cellular_automata_3.json");
	CellularAutomata cellular_automata(input);
}

TEST(CellularAutomata, ConstructorInvalidJson) {
	std::string input_path = "../test/resources/cellular_automata_invalid_json.json";
	EXPECT_THROW(CellularAutomataInput input(input_path), std::system_error);
}

TEST(CellularAutomata, ConstructorInvalidSchema) {
	std::string input_path = "../test/resources/cellular_automata_invalid_schema.json";
	EXPECT_THROW(CellularAutomataInput input(input_path), std::system_error);
}

TEST(CellularAutomata, Grow) {
	CellularAutomataInput input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
	cellular_automata.Grow();
}

TEST(CellularAutomata, UpdateWetCells) {
	CellularAutomataInput input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
	cellular_automata.Grow();
	cellular_automata.UpdateWetCells();
}

TEST(CellularAutomata, WriteResults) {
	CellularAutomataInput input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
	cellular_automata.WriteResults();
}

TEST(CellularAutomata, Run) {
	CellularAutomataInput input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
	cellular_automata.Run();
}
