#include <iostream>
#include <fstream>
#include <cstdlib>
#include "error.h"

namespace {
	const size_t SUCCESS = 0;
	const size_t ERROR_IN_COMMAND_LINE = 1;
	const size_t ERROR_IN_GDAL = 2;
	const size_t ERROR_UNHANDLED_EXCEPTION = 3;
}

void PrintErrorAndExit(const std::string& error_message, int exit_code) {
	std::cerr << "Error: " << error_message << std::endl;
	std::exit(exit_code);
}

void PrintWarning(const std::string& warning_message) {
	std::cerr << "Warning: " << warning_message << std::endl;
}

void CPLErrChk(CPLErr errnum) {
	if (errnum == CE_Failure) {
		std::cerr << CPLGetLastErrorMsg() << std::endl;
		exit(ERROR_IN_GDAL);
	}
}
