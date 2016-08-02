#include "sinks.h"

Sinks::Sinks(const rapidjson::Value& root, const Constants& constants) :
  ISinks(root, constants) { }
