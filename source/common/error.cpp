#include <iostream>
#include <fstream>
#include <cstdlib>
#include "error.h"

void CPLErrChk(CPLErr errnum) {
	if (errnum == CE_Failure || errnum == CE_Fatal) {
		std::string error_string = CPLGetLastErrorMsg();
		throw std::system_error(std::error_code(), error_string);
	}
}
