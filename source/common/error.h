#pragma once

#include <string>
#include <cpl_conv.h>

void PrintErrorAndExit(const std::string& error_message, int error_code = 1);
void PrintWarning(const std::string& warning_message);
void CPLErrChk(CPLErr errnum);
