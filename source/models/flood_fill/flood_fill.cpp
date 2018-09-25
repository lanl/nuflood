#include <omp.h>
#include <iostream>
#include <common/index_table.h>
#include "flood_fill.h"
#include "input.hpp"

FloodFill::FloodFill(const Input& input) {
	input_ = &input;
	num_seeds_ = num_wet_ = num_iterations_ = 0;
	B_.Read(input.elevation_path());

	if (!input.depth_path().empty()) {
		h_.Read(input.depth_path());
		w_.CopyFrom(B_);
		w_.Add(h_);
	} else if (!input.wse_path().empty()) {
		w_.Read(input.wse_path());
		h_.CopyFrom(w_);
		h_.Subtract(B_);
	} else {
		w_.CopyFrom(B_);
		h_.CopyFrom(B_);
		h_.Fill((prec_t)0);
	}

	for (auto it : input.point_sources_depth()) {
		INT_TYPE ij = h_.index(it.x, it.y);
		prec_t h_value = it.value > (prec_t)0 ? it.value : (prec_t)0;
		h_.SetAtIndex(ij, h_value);
		w_.SetAtIndex(ij, B_.GetFromIndex(ij) + h_value);
	}

	for (auto it : input.point_sources_wse()) {
		INT_TYPE ij = w_.index(it.x, it.y);
		prec_t w_value = it.value > B_.GetFromIndex(ij) ? it.value : B_.GetFromIndex(ij);
		w_.SetAtIndex(ij, w_value);
		h_.SetAtIndex(ij, w_value - B_.GetFromIndex(ij));
	}

	#pragma omp parallel for
	for (INT_TYPE ij = 0; ij < B_.num_pixels(); ij++) {
		if (B_.GetFromIndex(ij) == B_.nodata()) {
			w_.SetAtIndex(ij, w_.nodata());
			h_.SetAtIndex(ij, h_.nodata());
		} else if (w_.GetFromIndex(ij) == w_.nodata()) {
			w_.SetAtIndex(ij, B_.GetFromIndex(ij));
			h_.SetAtIndex(ij, 0.0);
		} else if (h_.GetFromIndex(ij) > 0.0) {
			#pragma omp critical
			{
				INT_TYPE i = ij / B_.width();
				INT_TYPE j = ij % B_.width();
				num_seeds_ += (INT_TYPE)seed_.Insert(i, j);
				num_wet_ += (INT_TYPE)wet_.Insert(i, j);
			}
		}
	}
}

void FloodFill::Grow(void) {
	seed_holder_.Clear();

	#pragma omp parallel
	#pragma omp single
	{
		for (auto it : seed_) {
			#pragma omp task firstprivate(it)
			for (const INT_TYPE& i : it.second) {
				const INT_TYPE j = it.first;
				prec_t w_ij = w_.GetFromIndices(i, j);
				if (w_ij == w_.nodata()) continue;

				for (INT_TYPE k = 0; k < 4; k++) {
					INT_TYPE ii = i + k % 2 - k / 2;
					INT_TYPE jj = j + (k + 1) % 2 - (k + 1)  / 3;
					if (ii < 0 || ii > B_.height() - 1 || jj < 0 || jj > B_.width() - 1) continue;

					prec_t B_ij = B_.GetFromIndices(ii, jj);
					if (B_ij == B_.nodata()) continue;

					prec_t h_ij = w_ij - B_ij;

					if (h_ij > 0.0 && !wet_.Contains(ii, jj)) {
						#pragma omp critical
						{
							seed_holder_.Insert(ii, jj);
							w_.SetAtIndices(ii, jj, w_ij);
							h_.SetAtIndices(ii, jj, h_ij);
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

	for (auto it : seed_holder_) {
		INT_TYPE j = it.first;
		for (const INT_TYPE& i : it.second) {
			num_seeds_ += (INT_TYPE)seed_.Insert(i, j);
			num_wet_ += (INT_TYPE)wet_.Insert(i, j);
		}
	}
}

void FloodFill::WriteResults(void) {
	if (!input_->output_depth_path().empty()) h_.Write(input_->output_depth_path());
	if (!input_->output_wse_path().empty()) w_.Write(input_->output_wse_path());
}

void FloodFill::Run(void) {
	while (num_seeds_ > 0) {
		FloodFill::Grow();
		FloodFill::UpdateWetCells();
		num_iterations_++;
	}

	FloodFill::WriteResults();
}

int main(int argc, char* argv[]) {
	// Check if a scenario file has been specified.
	if (argc <= 1) {
		PrintErrorAndExit("Scenario file has not been specified.");
	}

	// Read in the input.
	Input input(argv[1]);

	// Set up the model.
	FloodFill flood_fill(input);

	// Run the model.
	flood_fill.Run();

	// Program executed successfully.
	return 0;
}
