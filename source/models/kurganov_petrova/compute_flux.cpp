#include <omp.h>
#include "compute_flux.h"

inline void ComputeHX(const prec_t  w,        const prec_t  hu,    const prec_t  hv,
                      const prec_t  wp1,      const prec_t  hup1,  const prec_t  hvp1,
                      const prec_t  sw,       const prec_t  shu,   const prec_t  shv,
                      const prec_t  swp1,     const prec_t  shup1, const prec_t  shvp1,
                      const prec_t  B,        const prec_t  Bp1,   const prec_t  Bp2,
                      const prec_t  B_ij,     const prec_t  Bp1_ij,
                      const prec_t  gravity,  const prec_t  kappa, const prec_t  epsilon,
                            prec_t* Hx0,            prec_t* Hx1,         prec_t* Hx2,
                            prec_t* max_speed) {
	prec_t wp, hup, hvp; // Western interfacial values for cell (i+1, j).
	prec_t wm, hum, hvm; // Eastern interfacial values for cell (i, j).

	computeInterface(w, hu, hv, wp1, hup1, hvp1, sw, shu, shv, swp1, shup1,
	                 shvp1, B, Bp1, Bp2, B_ij, Bp1_ij, kappa, &wp, &hup, &hvp,
	                 &wm, &hum, &hvm);

	prec_t Fp0, Fp1, Fp2, app, apm;
	computeF(wp, Bp1, gravity, kappa, epsilon, &hup, &hvp, &app, &apm, &Fp0, &Fp1, &Fp2);

	prec_t Fm0, Fm1, Fm2, amp, amm;
	computeF(wm, Bp1, gravity, kappa, epsilon, &hum, &hvm, &amp, &amm, &Fm0, &Fm1, &Fm2);

	prec_t ap = fmaxf(fmaxf(app, amp), (prec_t)0);
	prec_t am = fminf(fminf(apm, amm), (prec_t)0);

	// Compute the eastern numerical flux for cell (i, j).
	if (ap-am > epsilon) {
		*Hx0 = computeH(wp,  Fp0, wm,  Fm0, am, ap);
		*Hx1 = computeH(hup, Fp1, hum, Fm1, am, ap);
		*Hx2 = computeH(hvp, Fp2, hvm, Fm2, am, ap);
		*max_speed = fmaxf(ap, -am);
	} else {
		*Hx0 = (prec_t)0;
		*Hx1 = (prec_t)0;
		*Hx2 = (prec_t)0;
		*max_speed = (prec_t)0;
	}
}

inline void ComputeHY(const prec_t  w,       const prec_t  hu,     const prec_t  hv,
                      const prec_t  wp1,     const prec_t  hup1,   const prec_t  hvp1,
                      const prec_t  sw,      const prec_t  shu,    const prec_t  shv,
                      const prec_t  swp1,    const prec_t  shup1,  const prec_t  shvp1,
                      const prec_t  B,       const prec_t  Bp1,    const prec_t  Bp2,
                      const prec_t  B_ij,    const prec_t  Bp1_ij,
                      const prec_t  gravity, const prec_t  kappa,  const prec_t  epsilon,
                            prec_t* Hy0,           prec_t* Hy1,          prec_t* Hy2,
                            prec_t* max_speed) {
	prec_t wp, hup, hvp; // Northern interfacial values for cell (i, j).
	prec_t wm, hum, hvm; // Southern interfacial values for cell (i+1, j).

	computeInterface(w, hu, hv, wp1, hup1, hvp1, sw, shu, shv, swp1, shup1,
	                 shvp1, B, Bp1, Bp2, B_ij, Bp1_ij, kappa, &wp, &hup, &hvp, &wm, &hum, &hvm);

	prec_t Gp0, Gp1, Gp2, bpp, bpm;
	computeG(wp, Bp1, gravity, kappa, epsilon, &hup, &hvp, &bpp, &bpm, &Gp0, &Gp1, &Gp2);

	prec_t Gm0, Gm1, Gm2, bmp, bmm;
	computeG(wm, Bp1, gravity, kappa, epsilon, &hum, &hvm, &bmp, &bmm, &Gm0, &Gm1, &Gm2);

	prec_t bp = fmaxf(fmaxf(bpp, bmp), (prec_t)0);
	prec_t bm = fminf(fminf(bpm, bmm), (prec_t)0);

	if (bp-bm > epsilon) {
		*Hy0 = computeH(wp,  Gp0, wm,  Gm0, bm, bp);
		*Hy1 = computeH(hup, Gp1, hum, Gm1, bm, bp);
		*Hy2 = computeH(hvp, Gp2, hvm, Gm2, bm, bp);
		*max_speed = fmaxf((*max_speed), fmaxf(bp, -bm));
	} else {
		*Hy0 = (prec_t)0;
		*Hy1 = (prec_t)0;
		*Hy2 = (prec_t)0;
	}
}

inline void ConstructHX(const Topography& B, const Conserved& U,
                        const Constants& C, const Slope& S, const INT_TYPE i,
                        const INT_TYPE j, Flux& H, const Time& T) {
	static prec_t* Bi = B.elevation_interpolated().data();

	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	static prec_t*  wx = S. wx().data();
	static prec_t* hux = S.hux().data();
	static prec_t* hvx = S.hvx().data();

	static prec_t*  H_wx = H. wx().data();
	static prec_t* H_hux = H.hux().data();
	static prec_t* H_hvx = H.hvx().data();

	INT_TYPE center = j * C.num_columns() + i;
	INT_TYPE   east = j * C.num_columns() + i+1;

	prec_t   B_ij = Bi[center];
	prec_t Bp1_ij = Bi[east  ];

	prec_t    B_west = (prec_t)0.5 * (B.elevation().Get(i,   j+1) + B.elevation().Get(i,   j));
	prec_t    B_east = (prec_t)0.5 * (B.elevation().Get(i+1, j+1) + B.elevation().Get(i+1, j));
	prec_t B_east_p1 = (prec_t)0.5 * (B.elevation().Get(i+2, j+1) + B.elevation().Get(i+2, j));

	ComputeHX(w [center], hu [center], hv [center],
             w [east  ], hu [east  ], hv [east  ],
	          wx[center], hux[center], hvx[center],
             wx[east  ], hux[east  ], hvx[east  ],
             B_west, B_east, B_east_p1, B_ij, Bp1_ij,
             C.gravitational_acceleration(), C.kappa(), C.machine_epsilon(),
	          &H_wx[center], &H_hux[center], &H_hvx[center],
	          T.max_step().GetPointer(i, j));
}

inline void ConstructHY(const Topography& B, const Conserved& U,
                        const Constants& C, const Slope& S, const INT_TYPE i,
                        const INT_TYPE j, Flux& H, const Time& T) {
	static prec_t* Bi = B.elevation_interpolated().data();

	static prec_t*  w = U.w ().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	static prec_t*  wy = S.wy ().data();
	static prec_t* huy = S.huy().data();
	static prec_t* hvy = S.hvy().data();

	static prec_t*  H_wy = H.wy ().data();
	static prec_t* H_huy = H.huy().data();
	static prec_t* H_hvy = H.hvy().data();

	INT_TYPE center = (j  ) * C.num_columns() + i;
	INT_TYPE north  = (j+1) * C.num_columns() + i;

	prec_t   B_ij = Bi[center];
	prec_t Bp1_ij = Bi[north ];

	prec_t    B_south = (prec_t)0.5 * (B.elevation().Get(i,   j) + B.elevation().Get(i+1,   j));
	prec_t    B_north = (prec_t)0.5 * (B.elevation().Get(i, j+1) + B.elevation().Get(i+1, j+1));
	prec_t B_north_p1 = (prec_t)0.5 * (B.elevation().Get(i, j+2) + B.elevation().Get(i+1, j+2));

	ComputeHY(w[center], hu[center], hv[center], w[north], hu[north], hv[north],
	          wy[center], huy[center], hvy[center], wy[north], huy[north],
             hvy[north], B_south, B_north, B_north_p1, B_ij, Bp1_ij,
	          C.gravitational_acceleration(), C.kappa(), C.machine_epsilon(),
	          &H_wy[center], &H_huy[center],  &H_hvy[center],
	          T.max_step().GetPointer(i, j));
}

inline void ApplyHX(const Flux& H, const Constants& C, const INT_TYPE i,
                    const INT_TYPE j, TimeDerivative& dUdt) {
	INT_TYPE center = j * C.num_columns() + i;
	INT_TYPE   west = j * C.num_columns() + i-1;

	static prec_t*  dUdt_w = dUdt.w ().data();
	static prec_t* dUdt_hu = dUdt.hu().data();
	static prec_t* dUdt_hv = dUdt.hv().data();

	static prec_t*  H_wx = H.wx ().data();
	static prec_t* H_hux = H.hux().data();
	static prec_t* H_hvx = H.hvx().data();

	dUdt_w [center] = (H_wx [west] - H_wx [center]) / C.cellsize_x();
	dUdt_hu[center] = (H_hux[west] - H_hux[center]) / C.cellsize_x();
	dUdt_hv[center] = (H_hvx[west] - H_hvx[center]) / C.cellsize_x();
}

inline void ApplyHY(const Flux& H, const Constants& C, const INT_TYPE i,
                    const INT_TYPE j, TimeDerivative& dUdt) {
	INT_TYPE center = (j  ) * C.num_columns() + i;
	INT_TYPE  south = (j-1) * C.num_columns() + i;

	static prec_t*  dUdt_w = dUdt. w().data();
	static prec_t* dUdt_hu = dUdt.hu().data();
	static prec_t* dUdt_hv = dUdt.hv().data();

	static prec_t*  H_wy = H. wy().data();
	static prec_t* H_huy = H.huy().data();
	static prec_t* H_hvy = H.hvy().data();

	dUdt_w [center] += (H_wy [south] - H_wy [center]) / C.cellsize_y();
	dUdt_hu[center] += (H_huy[south] - H_huy[center]) / C.cellsize_y();
	dUdt_hv[center] += (H_hvy[south] - H_hvy[center]) / C.cellsize_y();
}


inline void ApplySource(const Topography& B, const Conserved& U,
                        ISinks& Rm, const Sources& R,
                        const Infiltration& I, const Constants& C,
                        const Time& T, const INT_TYPE i, const INT_TYPE j,
                        TimeDerivative& dUdt) {
	INT_TYPE center = j * C.num_columns() + i;

	static prec_t* Bi = B.elevation_interpolated().data();

	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	static prec_t*  dUdt_w = dUdt. w().data();
	static prec_t* dUdt_hu = dUdt.hu().data();
	static prec_t* dUdt_hv = dUdt.hv().data();

	static prec_t* rainfall_grid = R.rainfall_grid().data();

	static prec_t* K_grid = I.K_grid().data();
	static prec_t* psi_grid = I.psi_grid().data();
	static prec_t* dtheta_grid = I.dtheta_grid().data();
	static prec_t* F = I.F().data();
	static prec_t* dF = I.dF().data();

	static prec_t K_value = I.K_value();
	static prec_t psi_value = I.psi_value();
	static prec_t dtheta_value = I.dtheta_value();

	prec_t h = w[center] - Bi[center];
	if (h < C.machine_epsilon()) {
		h = (prec_t)0;
		w [center] = Bi[center];
		hu[center] = (prec_t)0;
		hv[center] = (prec_t)0;
	}

	// Usually, source terms related to discharge are defined using h.
	// If a discharge source term is added which does not depend on h,
	// the code below should be reorganized.
	prec_t S1 = (prec_t)0;
	prec_t S2 = (prec_t)0;

	if (h > (prec_t)0) {
		// Compute the hv source terms
		prec_t B_east = (prec_t)0.5 * (B.elevation().Get(i+1, j) + B.elevation().Get(i+1, j+1));
		prec_t B_west = (prec_t)0.5 * (B.elevation().Get(i, j) + B.elevation().Get(i, j+1));
		S1 += C.gravitational_acceleration() * h * (B_west - B_east) / C.cellsize_x();

		// Compute the hv source terms
		prec_t B_north = (prec_t)0.5 * (B.elevation().Get(i, j+1) + B.elevation().Get(i+1, j+1));
		prec_t B_south = (prec_t)0.5 * (B.elevation().Get(i, j) + B.elevation().Get(i+1, j));
		S2 += C.gravitational_acceleration() * h * (B_south - B_north) / C.cellsize_y();
	}

	dUdt_hu[center] += S1;
	dUdt_hv[center] += S2;

	prec_t S0 = (prec_t)0;
	INT_TYPE k;

	// TODO: Use a better algorithm to find cells that have a corresponding
	// source. The for loop here could get computationally expensive if the
	// number of point sources is large.
	for (k = 0; k < R.points().size(); k++) {
		INT_TYPE x_index = R.points()[k].x_index(B.elevation_interpolated());
		INT_TYPE y_index = R.points()[k].y_index(B.elevation_interpolated());

		if (x_index == i && y_index == j) {
			S0 += R.points()[k].interpolated_rate(T.current());
		}
	}

	if (rainfall_grid != nullptr) {
		if (R.storm_curve_proportion()*rainfall_grid[center] > (prec_t)0) {
			prec_t rainfall_rate = rainfall_grid[center] * R.storm_curve_proportion() / T.step();
			S0 += rainfall_rate;
		}
	}

	// Apply infiltration via the explicit Green-Ampt model.
	if (K_value > (prec_t)0 || K_grid != nullptr) {
		prec_t K = K_grid != nullptr ? K_grid[center] : K_value;
		prec_t psi = psi_grid != nullptr ? psi_grid[center] : psi_value;
		prec_t dtheta = dtheta_grid != nullptr ? dtheta_grid[center] : dtheta_value;

		prec_t p1 = K*T.step() - (prec_t)(2) * F[center];
		prec_t p2 = K * (F[center] + psi*dtheta);
		prec_t inf_rate = (p1 + sqrtf(p1*p1 + (prec_t)(8)*p2*T.step())) / ((prec_t)(2)*T.step());

		inf_rate = (inf_rate > h / T.step()) ? h / T.step() : inf_rate;
		inf_rate = (inf_rate > (prec_t)(0)) ? inf_rate : (prec_t)(0);
		dF[center] = inf_rate;

		S0 -= inf_rate;
	}

	for (k = 0; k < Rm.points().size(); k++) {
		INT_TYPE x_index = Rm.points()[k].x_index(B.elevation_interpolated());
		INT_TYPE y_index = Rm.points()[k].y_index(B.elevation_interpolated());

		if (x_index == i && y_index == j) {
			if (h + S0*T.step() > (prec_t)0) { // If the anticipated water depth at t+1 is greater than zero...
				prec_t depth = Rm.points()[k].depth();

				if (h + S0 * T.step() > Rm.points()[k].rate() * T.step()) { // If the expected water depth is greater than the amount that can be drained...
					depth += Rm.points()[k].rate() * T.step(); // Integrate the drainage rate and add to the sink depth.
					S0 = S0 - Rm.points()[k].rate(); // Include the drainage rate in the source term.
				} else { // If the rate of water coming in is less than or equal to the drainage rate...
					depth += h + S0*T.step(); // The sink will accumulate all this water.
					S0 = -h / T.step(); // The sink will pick up any excess left behind at a rate of that excess.
				}

				Rm.points()[k].set_depth(depth);
			}
		}
	}

	dUdt_w[center] += S0;
}

inline void ConstructSlopes(const Topography& B, const Conserved& U,
                            const Constants& C, const INT_TYPE i,
                            const INT_TYPE j, Slope& S) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	static prec_t*  wx = S. wx().data();
	static prec_t* hux = S.hux().data();
	static prec_t* hvx = S.hvx().data();

	static prec_t*  wy = S. wy().data();
	static prec_t* huy = S.huy().data();
	static prec_t* hvy = S.hvy().data();

	static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE center = (j  ) * C.num_columns() + i;
	INT_TYPE   east = (j  ) * C.num_columns() + i+1;
	INT_TYPE   west = (j  ) * C.num_columns() + i-1;
	INT_TYPE  north = (j+1) * C.num_columns() + i;
	INT_TYPE  south = (j-1) * C.num_columns() + i;

	prec_t h_center = w[center] - Bi[center];
	prec_t   h_east = w[east  ] - Bi[east  ];
	prec_t   h_west = w[west  ] - Bi[west  ];
	prec_t  h_north = w[north ] - Bi[north ];
	prec_t  h_south = w[south ] - Bi[south ];

	prec_t u_center = computeVelocity(h_center, hu[center], C.kappa());
	prec_t   u_east = computeVelocity(h_east,   hu[east  ], C.kappa());
	prec_t   u_west = computeVelocity(h_west,   hu[west  ], C.kappa());
	prec_t  u_north = computeVelocity(h_north,  hu[north ], C.kappa());
	prec_t  u_south = computeVelocity(h_south,  hu[south ], C.kappa());

	prec_t v_center = computeVelocity(h_center, hv[center], C.kappa());
	prec_t   v_east = computeVelocity(h_east,   hv[east  ], C.kappa());
	prec_t   v_west = computeVelocity(h_west,   hv[west  ], C.kappa());
	prec_t  v_north = computeVelocity(h_north,  hv[north ], C.kappa());
	prec_t  v_south = computeVelocity(h_south,  hv[south ], C.kappa());

	wx [center] = limiter(w[west], w[center], w[east]);
	hux[center] = limiter(u_west,  u_center,  u_east );
	hvx[center] = limiter(v_west,  v_center,  v_east );

	wy [center] = limiter(w[south], w[center], w[north]);
	huy[center] = limiter(u_south,  u_center,  u_north );
	hvy[center] = limiter(v_south,  v_center,  v_north );
}

void ComputeFlux(const Topography& B, ISinks& Rm, const Sources& R,
                 const Infiltration& I, const Constants& C, Conserved& U,
                 Slope& S, Flux& H, TimeDerivative& dUdt, const Time& T,
                 ActiveCells& A) {
	if (A.Tracking()) {
		Map::const_iterator it;

		#pragma omp parallel default(shared) private(it)
		for (it = A.active_.begin(); it != A.active_.end(); ++it) {
			const INT_TYPE& j = it->first;
			#pragma omp single nowait
			for (const INT_TYPE& i: it->second) {
				if (i > 0 && i < C.num_columns() - 1 && j > 0 && j < C.num_rows() - 1) {
					ConstructSlopes(B, U, C, i, j, S);
				}
			}
		}

		#pragma omp parallel default(shared) private(it)
		for (it = A.active_.begin(); it != A.active_.end(); ++it) {
			const INT_TYPE& j = it->first;
			#pragma omp single nowait
			for (const INT_TYPE& i: it->second) {
				if (i > 0 && i < C.num_columns() - 2 && j > 0 && j < C.num_rows() - 2) {
					ConstructHX(B, U, C, S, i, j, H, T);
					ConstructHY(B, U, C, S, i, j, H, T);
				}
			}
		}

		#pragma omp parallel default(shared) private(it)
		for (it = A.active_.begin(); it != A.active_.end(); ++it) {
			const INT_TYPE& j = it->first;
			#pragma omp single nowait
			for (const INT_TYPE& i: it->second) {
				if (i > 1 && i < C.num_columns() - 2 && j > 1 && j < C.num_rows() - 2) {
					ApplyHX(H, C, i, j, dUdt);
					ApplyHY(H, C, i, j, dUdt);
				}
			}
		}

		#pragma omp parallel default(shared) private(it)
		for (it = A.active_.begin(); it != A.active_.end(); ++it) {
			const INT_TYPE& j = it->first;
			#pragma omp single nowait
			for (const INT_TYPE& i: it->second) {
				if (i > 1 && i < C.num_columns() - 2 && j > 1 && j < C.num_rows() - 2) {
					ApplySource(B, U, Rm, R, I, C, T, i, j, dUdt);
				}
			}
		}
	} else {
		// This constructs slopes for the inner cells
		INT_TYPE i, j;
		#pragma omp parallel for default(shared) private(i, j)
		for (j = 1; j < C.num_rows() - 1; j++) {
			for (i = 1; i < C.num_columns() - 1; i++) {
				ConstructSlopes(B, U, C, i, j, S);
			}
		}

		// This constructs numerical flux vectors at the eastern and northern edges
		// of cell (i, j). It requires information from the cell east/north of it (as
		// well as bathymetry values two nodes east/north of the cell's
		// western/southern interface).
		#pragma omp parallel for default(shared) private(i, j)
		for (j = 1; j < C.num_rows() - 2; j++) {
			for (i = 1; i < C.num_columns() - 2; i++) {
				ConstructHX(B, U, C, S, i, j, H, T);
				ConstructHY(B, U, C, S, i, j, H, T);
			}
		}

		// To apply the numerical flux vectors to dUdt, we need numerical flux
		// information at the top/right interface of cell (i, j). Flux information
		// doesn't exist at i = 0 or j = 0, and eastern flux information doesn't
		// exist for i = *(C->num_columns)-1 or j = *(C->num_rows)-1, so we can't
		// apply fluxes near these regions.
		#pragma omp parallel for default(shared) private(i, j)
		for (j = 2; j < C.num_rows() - 2; j++) {
			for (i = 2; i < C.num_columns() - 2; i++) {
				ApplyHX(H, C, i, j, dUdt);
				ApplyHY(H, C, i, j, dUdt);
			}
		}

		// This applies source terms to the time derivatives.
		#pragma omp parallel for default(shared) private(i, j)
		for (j = 2; j < C.num_rows() - 2; j++) {
			for (i = 2; i < C.num_columns() - 2; i++) {
				ApplySource(B, U, Rm, R, I, C, T, i, j, dUdt);
			}
		}
	}
}
