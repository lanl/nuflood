#include <omp.h>
#include <math.h>
#include <iterator>
#include "compute_timestep.h"

void ComputeTimestep(ActiveCells& A, const Constants& C, Time& T) {
	prec_t max_val = (prec_t)0;

	if (A.Tracking()) {
		Map::const_iterator it;
		#pragma omp parallel default(shared) private(it) reduction(max : max_val)
		for (it = A.active_.begin(); it != A.active_.end(); ++it) {
			const INT_TYPE& j = it->first;
			#pragma omp single nowait
			for (const INT_TYPE& i: it->second) {
				if (i > 1 && i < C.num_columns() - 2 && j > 1 && j < C.num_rows() - 2) {
					if (T.max_step().Get(i, j) > max_val) {
						max_val = T.max_step().Get(i, j);
					}
				}
			}
		}
	} else {
		INT_TYPE i, j;
		#pragma omp parallel for default(shared) private(i, j) reduction(max : max_val)
		for (j = 2; j < C.num_rows() - 2; j++) {
			for (i = 2; i < C.num_columns() - 2; i++) {
				if (T.max_step().Get(i, j) > max_val) {
					max_val = T.max_step().Get(i, j);
				}
			}
		}
	}

	T.set_step(C.cellsize_x() / fmaxf(fmaxf(4.0f*max_val, C.kappa()), 10.0f));
}
