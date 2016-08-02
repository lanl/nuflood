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
				// Both coordinates should be in decimal degrees.
				double x = points_json[i]["x"].GetDouble();
				double y = points_json[i]["y"].GetDouble();

				// Hydrograph is read in assuming units of (s, m^3 / s).
				std::string path = points_json[i]["hydrograph"].GetString();
				PointSource<prec_t> point_source(x, y, File(path));

				// Convert rate from m^3 / s to m / s.
				point_source.Scale((prec_t)(1.0 / constants.cell_area()));
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
				// Assumes units of meters.
				rainfall_grid_.Load(rainfall_grid_file);
			}

			File storm_curve_file;
			ReadParameter(rainfall_json, "stormCurve", storm_curve_file);

			if (!storm_curve_file.IsEmpty()) {
				// Assumes unitless time series.
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
