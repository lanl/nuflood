#include <iostream>
#include "cellular_automata.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

CellularAutomata::CellularAutomata(const CellularAutomataInput& input) {
	input_ = &input;
	num_seeds_ = num_wet_ = 0;
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
		h_.CopyFrom(w_);
		h_.Fill((prec_t)0);
	}

	for (auto it : input.point_sources_depth()) {
		int_t ij = h_.index(it.x, it.y);
		prec_t h_value = it.value > (prec_t)0 ? it.value : (prec_t)0;
		h_.SetAtIndex(ij, h_value);
		w_.SetAtIndex(ij, B_.GetFromIndex(ij) + h_value);
	}

	for (auto it : input.point_sources_wse()) {
		int_t ij = w_.index(it.x, it.y);
		prec_t w_value = it.value > B_.GetFromIndex(ij) ? it.value : B_.GetFromIndex(ij);
		w_.SetAtIndex(ij, w_value);
		h_.SetAtIndex(ij, w_value - B_.GetFromIndex(ij));
	}

	#pragma omp parallel for
	for (int_t ij = 0; ij < B_.num_pixels(); ij++) {
		if (B_.GetFromIndex(ij) == B_.nodata()) {
			w_.SetAtIndex(ij, w_.nodata());
			h_.SetAtIndex(ij, h_.nodata());
		} else if (w_.GetFromIndex(ij) == w_.nodata()) {
			w_.SetAtIndex(ij, B_.GetFromIndex(ij));
			h_.SetAtIndex(ij, (prec_t)0);
		} else if (h_.GetFromIndex(ij) > (prec_t)0) {
			#pragma omp critical
			{
				int_t i = ij / B_.width();
				int_t j = ij % B_.width();
				num_seeds_ += (int_t)seed_.Insert(i, j);
				num_wet_ += (int_t)wet_.Insert(i, j);
			}
		}
	}
}

void CellularAutomata::Grow(void) {
	seed_holder_.Clear();

	#pragma omp parallel
	#pragma omp single
	{
		for (auto it : seed_) {
			#pragma omp task firstprivate(it)
			for (const int_t& i : it.second) {
				const int_t j = it.first;
				prec_t w_ij = w_.GetFromIndices(i, j);
				if (w_ij == w_.nodata()) continue;

				const int i_shift[4] = {-1, 1, 0, 0};
				const int j_shift[4] = {0, 0, -1, 1};

				for (int_t k = 0; k < 4; k++) {
					int_t ii = i + i_shift[k];
					if (ii < 0 || ii > B_.height() - 1) continue;

					int_t jj = j + j_shift[k];
					if (jj < 0 || jj > B_.width() - 1) continue;

					prec_t B_ij = B_.GetFromIndices(ii, jj);
					if (B_ij == B_.nodata()) continue;

					prec_t h_ij = w_ij - B_ij;

					if (h_ij > (prec_t)0 && !wet_.Contains(ii, jj)) {
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

void CellularAutomata::UpdateWetCells(void) {
	seed_.Clear();
	num_seeds_ = 0;

	for (auto it : seed_holder_) {
		int_t j = it.first;

		for (const int_t& i : it.second) {
			num_seeds_ += (int_t)seed_.Insert(i, j);
			num_wet_ += (int_t)wet_.Insert(i, j);
		}
	}
}

void CellularAutomata::WriteResults(void) {
	if (!input_->output_depth_path().empty()) h_.Write(input_->output_depth_path());
	if (!input_->output_wse_path().empty()) w_.Write(input_->output_wse_path());
}

void CellularAutomata::Run(void) {
	while (num_seeds_ > 0) {
		CellularAutomata::Grow();
		CellularAutomata::UpdateWetCells();
	}

	CellularAutomata::WriteResults();
}
