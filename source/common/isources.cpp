#include <iostream>
#include "parameter.h"
#include "isources.h"

ISources::ISources(const rapidjson::Value& root, const IConstants& constants) {
	if (root.HasMember("sources")) {
		const rapidjson::Value& source_json = root["sources"];

		if (source_json.HasMember("points")) {
			const rapidjson::Value& points_json = source_json["points"];
			assert(points_json.IsArray());
			for (rapidjson::SizeType i = 0; i < points_json.Size(); i++) {
				double x = points_json[i]["x"].GetDouble();
				double y = points_json[i]["y"].GetDouble();
				std::string path = points_json[i]["hydrograph"].GetString();
				PointSource<prec_t> point_source(x, y, File(path));
				prec_t cfs_to_m_per_s = (prec_t)((1.0 / 35.31467) / (constants.cellsize_x()*constants.cellsize_y()));
				point_source.Scale(cfs_to_m_per_s);
				points_.push_back(point_source);
			}
		}

		File marigram_file;
		ReadParameter(source_json, "marigram", marigram_file);
		if (!marigram_file.IsEmpty()) {
			marigram_.Load(marigram_file);
		}

		storm_curve_proportion_ = (prec_t)0;
		if (source_json.HasMember("rainfall")) {
			const rapidjson::Value& rainfall_json = source_json["rainfall"];

			File rainfall_grid_file;
			ReadParameter(rainfall_json, "grid", rainfall_grid_file);
			if (!rainfall_grid_file.IsEmpty()) {
				rainfall_grid_.Load(rainfall_grid_file);
				rainfall_grid_.Scale((prec_t)0.0254); // Scale from inches to meters.
			}

			File storm_curve_file;
			ReadParameter(rainfall_json, "stormCurve", storm_curve_file);
			if (!storm_curve_file.IsEmpty()) {
				storm_curve_.Load(storm_curve_file);
			}
		}
	}
}

void ISources::Update(const ITime& T) {
	if (!storm_curve_.IsEmpty()) {
		if (T.current() + T.step() <= storm_curve_.end_time()) {
			storm_curve_proportion_ = storm_curve_.interpolated_value(T.current() + T.step()) -
											  storm_curve_.interpolated_value(T.current());
		} else {
			storm_curve_proportion_ = (prec_t)(0);
		}
	}
}
