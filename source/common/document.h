#pragma once

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/filereadstream.h>
#include "file.h"

class Document {
public:
	Document(const File& file);
	Document(const std::string path);
	rapidjson::Document root;
};
