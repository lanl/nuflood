#include "error.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

void PrintErrorAndExit(const std::string &error_message, int exit_code) {
    std::cerr << "Error: " << error_message << std::endl;
    std::exit(exit_code);
}

void PrintWarning(const std::string &warning_message) {
    std::cerr << "Warning: " << warning_message << std::endl;
}
