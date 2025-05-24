#pragma once

#include <string>

void PrintErrorAndExit(const std::string &error_message, int exit_code = 1);
void PrintWarning(const std::string &warning_message);
