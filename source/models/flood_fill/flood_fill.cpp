#include <omp.h>
#include <iostream>
#include <common/parameter.h>
#include <common/index_table.h>
#include <common/file.h>
#include "flood_fill.h"

FloodFill::FloodFill(const rapidjson::Value& root) {
	File depth_file_;
	File bathymetry_file_;
	File water_surface_elevation_file_;
	float added_depth = 0.0f;
	output_folder_.Clear();

	ReadParameter(root, "depthFile", depth_file_);
	ReadParameter(root, "bathymetryFile", bathymetry_file_);
	ReadParameter(root, "waterSurfaceElevationFile", water_surface_elevation_file_);
	ReadParameter(root, "outputFolder", output_folder_);
	ReadParameter(root, "addedDepth", added_depth); // added depth (meters)

	B_.Load(bathymetry_file_);
	B_.set_name("bathymetry");

	if (!water_surface_elevation_file_.IsEmpty()) {
		w_.Load(water_surface_elevation_file_);
	} else {
		w_.Copy(B_);
	}

	w_.set_name("waterSurfaceElevation");

	if (root.HasMember("seeds")) {
		const rapidjson::Value& point_array_json = root["seeds"];

		unsigned int i = 0;
		while (i < point_array_json.Size()) {
			double x = point_array_json[i++].GetDouble();
			double y = point_array_json[i++].GetDouble();
			double z = point_array_json[i++].GetDouble();
			w_.Set(z, x, y);
		}
	}

	if (!depth_file_.IsEmpty()) {
		h_.Load(depth_file_);
		w_.Add(h_);
	} else {
		h_.Copy(B_);
		h_.Fill(0.0f);
	}

	h_.set_name("depth");

	if (!w_.DimensionsMatch(B_) || !h_.DimensionsMatch(B_)) {
		PrintErrorAndExit("Input grid files do not have matching dimensions.");
	}

	num_iterations_ = 0;
	num_seeds_ = 0;
	num_wet_ = 0;

	#pragma omp parallel for
	for (INT_TYPE j = 0; j < B_.num_rows(); j++) {
		for (INT_TYPE i = 0; i < B_.num_columns(); i++) {
			float wse = w_.Get(i, j);

			bool skip = false;
			if (wse == w_.nodata_value() || B_.Get(i, j) == B_.nodata_value()) {
				skip = true;
			}

			h_.Set(wse - B_.Get(i, j), i, j);

			if (h_.Get(i, j) > 0.0f && !skip) {
				if (added_depth > 0.0f) {
					h_.Set(added_depth + wse - B_.Get(i, j), i, j);
					w_.Set(added_depth + wse, i, j);
				}

				#pragma omp critical
				{
					num_seeds_ += seed_.Insert(i, j) ? 1 : 0;
					num_wet_ += wet_.Insert(i, j) ? 1 : 0;
				}
			} else {
				h_.Set(0.0f, i, j);
				w_.Set(B_.Get(i, j), i, j);
			}
		}
	}
}

void FloodFill::Grow(void) {
	seed_holder_.Clear();

	#pragma omp parallel
	#pragma omp single
	{
		for (Map::const_iterator it = seed_.begin(); it != seed_.end(); ++it) {
			#pragma omp task firstprivate(it)
			for (const INT_TYPE& column: it->second) {
				const INT_TYPE row = it->first;
				bool out_of_bounds = row == 0 || row == B_.num_rows()-1 ||
				                     column == 0 || column == B_.num_columns()-1;
				if (out_of_bounds) {
					continue;
				}

				if (w_.Get(column, row) == w_.nodata_value()) {
					continue;
				}

				float wij = w_.Get(column, row);
				INT_TYPE i, j;
				for (INT_TYPE m = 0; m < 4; m++) {
					switch(m) {
						case 0:
							i = column + 1;
							j = row;
							break;
						case 1:
							i = column - 1;
							j = row;
							break;
						case 2:
							i = column;
							j = row + 1;
							break;
						case 3:
							i = column;
							j = row - 1;
							break;
					}

					if (B_.Get(i, j) == B_.nodata_value()) {
						continue;
					}

					float Bij = B_.Get(i, j);
					float hij = wij - Bij;

					if (hij > 0.0f && !wet_.Contains(i, j)) {
						#pragma omp critical
						{
							seed_holder_.Insert(i, j);
							w_.Set(wij, i, j);
							h_.Set(hij, i, j);
						}
					}
				}
			}
		}

		#pragma omp taskwait
	}
}

void FloodFill::UpdateWetCells(void) {
	seed_.Clear();
	num_seeds_ = 0;

	for (Map::const_iterator it = seed_holder_.begin(); it != seed_holder_.end(); ++it) {
		INT_TYPE row = it->first;
		for (const INT_TYPE& column: it->second) {
			num_seeds_ += seed_.Insert(column, row) ? 1 : 0;
			num_wet_ += wet_.Insert(column, row) ? 1 : 0;
		}
	}
}

void FloodFill::FillCorner(INT_TYPE column, INT_TYPE row) {
	INT_TYPE i, j;
	for (INT_TYPE m = 0; m < 4; m++) {
		switch(m) {
			case 0:
				i = column + 1;
				j = row;
				break;
			case 1:
				i = column - 1;
				j = row;
				break;
			case 2:
				i = column;
				j = row + 1;
				break;
			case 3:
				i = column;
				j = row - 1;
				break;
		}

		if (i < 0 || i > B_.num_columns() - 1 || 
		    j < 0 || j > B_.num_rows() - 1 ||
			 B_.Get(i, j) == B_.nodata_value() ||
			 h_.Get(i, j) <= 0.0) {
			continue;
		}

		if (h_.Get(i, j) > 0.0) {
			float h_new = w_.Get(i, j) - B_.Get(column, row);
			h_.Set(h_new, column, row);
			break;
		}
	}
}

void FloodFill::FillCorners(void) {
	FillCorner(0, 0);
	FillCorner(0, B_.num_rows() - 1);
	FillCorner(B_.num_columns() - 1, 0);
	FillCorner(B_.num_columns() - 1, B_.num_rows() - 1);
}

void FloodFill::Run(void) {
	std::cout << "Iteration" << "\t" << "Number of seeds" << std::endl;
	std::cout << num_iterations_ << "\t" << num_seeds_ << std::endl;

	while (num_seeds_ > 0) {
		Grow();
		UpdateWetCells();
		num_iterations_++;
		std::cout << num_iterations_ << "\t" << num_seeds_ << std::endl;
	}

	FillCorners();
	//ReduceEdges();
	h_.Write(output_folder_);
}

void FloodFill::ReduceEdges(void) {
	Grid<float> h_tmp;
	h_tmp.Copy(h_);

	#pragma omp parallel for
	for (INT_TYPE j = 0; j < B_.num_rows(); j++) {
		for (INT_TYPE i = 0; i < B_.num_columns(); i++) {
			bool left_dry = i == 0 ? false : h_tmp.Get(i - 1, j) <= 0.0;
			bool right_dry = i == B_.num_columns() - 1 ? false : h_tmp.Get(i + 1, j) <= 0.0;
			bool top_dry = j == B_.num_rows() - 1 ? false : h_tmp.Get(i, j + 1) <= 0.0;
			bool bottom_dry = j == 0 ? false : h_tmp.Get(i, j - 1) <= 0.0;
			bool any_dry = left_dry || right_dry || top_dry || bottom_dry;

			if (any_dry) {
				h_.Set(0.0f, i, j);
			}
		}
	}
}

int main(int argc, char* argv[]) {
	// Check if a scenario file has been specified.
	if (argc <= 1) {
		PrintErrorAndExit("Scenario file has not been specified.");
	}

	// Read the scenario file.
	File scenario(argv[1]);

	// Parse the scenario file as a JSON document.
	Document json(scenario);

	// Set up the model.
	FloodFill model(json.root["parameters"]);

	// Run the model.
	model.Run();

	// Program executed successfully.
	return 0;
}
