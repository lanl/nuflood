#pragma once

#include "file.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

class Document {
  public:
    Document(const File &file);
    rapidjson::Document root;
};
