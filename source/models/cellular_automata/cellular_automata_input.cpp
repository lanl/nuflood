#include "cellular_automata_input.h"
#include "point_source.h"

//! Constructor for CellularAutomataInput.
/*! \param path Path to the JSON document defining model parameters.
 * The JSON must conform to the <a href="cellularAutomata.schema.json">JSON
 * Schema</a>. */
CellularAutomataInput::CellularAutomataInput(std::string path) {
	// Create the schema document object.
	rapidjson::Document schema_document;
	schema_document.Parse(CELLULARAUTOMATA_SCHEMA);
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
			std::string error_string = "Scenario file '" + path + "' is not a valid JSON file.";
			throw std::system_error(std::error_code(), error_string);
		} else {
			std::string error_string = "Scenario file '" + path + "' did not pass schema validation.";
			throw std::system_error(std::error_code(), error_string);
		}
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
				PointSource point_source{x, y, value};
				point_sources_depth_.push_back(point_source);
			} else if (point_list_data[i].HasMember("waterSurfaceElevation")) {
				double value = point_list_data[i]["waterSurfaceElevation"].GetDouble(); 
				PointSource point_source{x, y, value};
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
