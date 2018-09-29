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

void CPLErrChk(CPLErr errnum) {
	if (errnum == CE_Failure) {
		std::string error_string = CPLGetLastErrorMsg();
		throw std::system_error(std::error_code(), error_string);
	}
}
