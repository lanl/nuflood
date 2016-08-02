#ifdef BIG_GRID
#define INT_TYPE long unsigned int
#else
#define INT_TYPE unsigned int
#endif

#include <math.h>
#include <omp.h>
#include "integrate.h"

inline void EulerStep(const Topography& B, const TimeDerivative& dUdt,
                      const Friction& Sf, const Time& time, const Constants& C,
                      const INT_TYPE center, Conserved& U, Infiltration& I) {
	static prec_t* Bi = B.elevation_interpolated().data();
	static prec_t* w = U.w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	static prec_t* w_old = U.w_old().data();
	static prec_t* dUdt_w = dUdt.w().data();
	static prec_t* dUdt_hu = dUdt.hu().data();
	static prec_t* dUdt_hv = dUdt.hv().data();
	static prec_t* h = U.h().data();
	static prec_t* h_max = U.h_max().data();
	static prec_t* q = U.q().data();
	static prec_t* q_max = U.q_max().data();
	static prec_t* F = I.F().data();
	static prec_t* F_old = I.F_old().data();
	static prec_t* dF = I.dF().data();

	// Save the previous water surface elevation.
	w_old[center] = w[center];

	// Compute the friction term.
	prec_t G = (prec_t)0;
	prec_t n = (Sf.manning_grid().data() != nullptr) ? Sf.manning_grid().data()[center] : Sf.manning_value();
	if (n > (prec_t)0) {
		prec_t h = w[center] - Bi[center];
		if (h > C.machine_epsilon()) {
			G = time.step() * C.gravitational_acceleration() * n*n *
			    powf((prec_t)(2.0)*h / (h*h + fmaxf(h*h, C.kappa() * C.kappa())), (prec_t)(7.0/3.0)) *
			    sqrtf(hu[center]*hu[center] + hv[center]*hv[center]);
		}
	}

	w [center] = (w [center] + dUdt_w [center] * time.step());
	hu[center] = (hu[center] + dUdt_hu[center] * time.step()) / (prec_t)(1.0 + G);
	hv[center] = (hv[center] + dUdt_hv[center] * time.step()) / (prec_t)(1.0 + G);

	// Correct quantities if the depth is less than epsilon.
	if (w[center] - Bi[center] < C.machine_epsilon()) {
		w [center] = Bi[center];
		hu[center] = (prec_t)0;
		hv[center] = (prec_t)0;

		// Correct derived quantities.
		if (U.h().data() != nullptr) U.h().data()[center] = (prec_t)0;
		if (U.q().data() != nullptr) U.q().data()[center] = (prec_t)0;
	} else {
		// Compute derived quantities.
		if (h != nullptr) h[center] = w[center] - Bi[center];
		if (q != nullptr) q[center] = sqrtf(hu[center]*hu[center] +	hv[center]*hv[center]);
		if (h_max != nullptr) h_max[center] = fmaxf(h_max[center], h[center]);
		if (q_max != nullptr) q_max[center] = fmaxf(q_max[center], q[center]);
	}

	// If infiltration is being modeled, integrate the infiltrated depth.
	if (F != nullptr) {
		F_old[center] = F[center];
		F[center] = F[center] + dF[center] * time.step();
	}
}

void Integrate(const Topography& B, const TimeDerivative& dUdt,
               const Friction& Sf, const Constants& C, const Time& T,
               Conserved& U, Infiltration& I, ActiveCells& A) {
	static prec_t* Bi = B.elevation_interpolated().data();
	static prec_t* w = U.w().data();
	static prec_t* w_old = U.w_old().data();

	if (A.Tracking()) {
		Map::const_iterator it;
		#pragma omp parallel default(shared) private(it)
		for (it = A.active_.begin(); it != A.active_.end(); ++it) {
			const INT_TYPE& j = it->first;
			#pragma omp single nowait
			for (const INT_TYPE& i: it->second) {
				if (i > 1 && i < C.num_columns() - 2 && j > 1 && j < C.num_rows() - 2) {
					const INT_TYPE center = j * C.num_columns() + i;
					EulerStep(B, dUdt, Sf, T, C, center, U, I);
					if (w[center] > w_old[center] && w_old[center] == Bi[center]) {
						#pragma omp critical
						{
							A.InsertWet(i, j);
						}
					}
				}
			}
		}
	} else {
		INT_TYPE i, j, center;
		#pragma omp parallel for default(shared) private(i, j, center)
		for (j = 2; j < C.num_rows() - 2; j++) {
			for (i = 2; i < C.num_columns() - 2; i++) {
				center = j * C.num_columns() + i;
				EulerStep(B, dUdt, Sf, T, C, center, U, I);
			}
		}
	}
}
