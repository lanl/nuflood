#pragma once

#include <iostream>
#include <rapidjson/filereadstream.h>
#include <rapidjson/document.h>
#include <rapidjson/schema.h>
#include <rapidjson/stringbuffer.h>
#include "schema.h"

struct PointSourceS {
	double x;
	double y;
	double value;
};

class Input {
public:
	// Constructor.
	Input(std::string path);

	// Getters.
	const std::string elevation_path(void) const { return elevation_path_; }
	const std::string depth_path(void) const { return depth_path_; }
	const std::string wse_path(void) const { return wse_path_; }
	const std::string output_depth_path(void) const { return output_depth_path_; }
	const std::string output_wse_path(void) const { return output_wse_path_; }
	const std::string output_summary_path(void) const { return output_summary_path_; }
	const std::vector<PointSourceS>& point_sources_depth(void) const { return point_sources_depth_; }
	const std::vector<PointSourceS>& point_sources_wse(void) const { return point_sources_wse_; }

protected:
	std::string elevation_path_;
	std::string depth_path_;
	std::string wse_path_; // wse := water surface elevation
	std::string output_depth_path_;
	std::string output_wse_path_;
	std::string output_summary_path_;
	std::vector<PointSourceS> point_sources_depth_;
	std::vector<PointSourceS> point_sources_wse_;
};

inline Input::Input(std::string path) {
	// Create the schema document object.
	rapidjson::Document schema_document;
	schema_document.Parse(FLOODFILL_SCHEMA);
	rapidjson::SchemaDocument schema(schema_document);

	char buffer[65536];
	FILE* p_file = fopen(path.c_str(), "r");
	rapidjson::FileReadStream input_stream(p_file, buffer, sizeof(buffer));
	rapidjson::SchemaValidatingReader<rapidjson::kParseDefaultFlags, rapidjson::FileReadStream,
	                                  rapidjson::UTF8<> > reader(input_stream, schema);

	rapidjson::Document document;
	document.Populate(reader);
	fclose(p_file);

	if (!reader.GetParseResult()) {
		if (reader.IsValid()) {
			std::cerr << "Scenario file '" + path + "' is not a valid JSON file." << std::endl;
		} else {
			std::cerr << "Scenario file '" + path + "' did not pass schema validation." << std::endl;
		}

		std::exit(1);
	}

	elevation_path_ = document["elevationPath"].GetString();

	if (document.HasMember("depthPath")) {
		depth_path_ = document["depthPath"].GetString();
	} else if (document.HasMember("waterSurfaceElevationPath")) {
		wse_path_ = document["waterSurfaceElevationPath"].GetString();
	}

	if (document.HasMember("pointSources")) {
		const rapidjson::Value& point_list_data = document["pointSources"];

		for (unsigned int i = 0; i < point_list_data.Size(); i++) {
			double x = point_list_data[i]["x"].GetDouble();
			double y = point_list_data[i]["y"].GetDouble();

			if (point_list_data[i].HasMember("depth")) {
				double value = point_list_data[i]["depth"].GetDouble();
				PointSourceS point_source{x, y, value};
				point_sources_depth_.push_back(point_source);
			} else if (point_list_data[i].HasMember("waterSurfaceElevation")) {
				double value = point_list_data[i]["waterSurfaceElevation"].GetDouble(); 
				PointSourceS point_source{x, y, value};
				point_sources_wse_.push_back(point_source);
			}
		}
	}

	if (document.HasMember("output")) {
		const rapidjson::Value& output_data = document["output"];

		if (output_data.HasMember("depthPath")) {
			output_depth_path_ = output_data["depthPath"].GetString();
		}

		if (output_data.HasMember("waterSurfaceElevationPath")) {
			output_wse_path_ = output_data["waterSurfaceElevationPath"].GetString();
		}

		if (output_data.HasMember("summaryPath")) {
			output_summary_path_ = output_data["summaryPath"].GetString();
		}
	}
}
