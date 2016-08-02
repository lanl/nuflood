#include <iostream>
#include "parameter.h"
#include "isinks.h"

ISinks::ISinks(const rapidjson::Value& root, const IConstants& constants) {
	if (root.HasMember("sinks")) {
		const rapidjson::Value& sinks_json = root["sinks"];

		if (sinks_json.HasMember("points")) {
			const rapidjson::Value& points_json = sinks_json["points"];
			assert(points_json.IsArray());
			for (rapidjson::SizeType i = 0; i < points_json.Size(); i++) {
				double x = points_json[i]["x"].GetDouble();
				double y = points_json[i]["y"].GetDouble();

				prec_t depth = (prec_t)0;
				if (points_json[i].HasMember("depth")) {
					depth = (prec_t)points_json[i]["depth"].GetDouble();
				}

				std::string name = "";
				if (points_json[i].HasMember("name")) {
					name = points_json[i]["name"].GetString();
				}

				// Drainage rate is read in assuming units of m^3 / s.
				prec_t rate = (prec_t)points_json[i]["rate"].GetDouble();
				PointSink<prec_t> point_sink(x, y, rate, depth, name);

				// Convert rate from m^3 / s to m / s.
				point_sink.Scale((prec_t)(1.0 / constants.cell_area()));
				points_.push_back(point_sink);
			}
		}
	}
}
