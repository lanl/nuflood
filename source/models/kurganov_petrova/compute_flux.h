#pragma once

#include <math.h>
#include <common/precision.h>
#include <common/isinks.h>
#include "topography.h"
#include "sources.h"
#include "infiltration.h"
#include "constants.h"
#include "conserved.h"
#include "slope.h"
#include "flux.h"
#include "time_derivative.h"
#include "time.h"
#include "active_cells.h"

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#define INLINE
#endif
#define THETA 1.3

void ComputeFlux(const Topography& B, ISinks& Rm, const Sources& R,
                 const Infiltration& I, const Constants& C, Conserved& U,
                 Slope& S, Flux& H, TimeDerivative& dUdt, const Time& T,
                 ActiveCells& A);

HOST DEVICE
inline prec_t fsignf(const prec_t x) {
	return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

HOST DEVICE
inline prec_t minmod(const prec_t a, const prec_t b, const prec_t c) {
	return 0.25f*fsignf(a)*(fsignf(a)+fsignf(b))*(fsignf(b)+fsignf(c))*
	             fminf(fminf(fabsf(a), fabsf(b)), fabsf(c));
}

HOST DEVICE
inline prec_t limiter(const prec_t back, const prec_t center, const prec_t front) {
	return 0.5f * minmod(THETA*(center-back), 0.5f*(front-back), THETA*(front-center));
}

HOST DEVICE
inline prec_t computeVelocity(const prec_t h, const prec_t hz, const prec_t kappa) {
	// If depth is less than value, correct velocity.
	return 2.0f*h*hz / (h*h + fmaxf(h*h, kappa*kappa));
}

HOST DEVICE
inline prec_t computeFrictionTerm(const prec_t h, const prec_t hz, const prec_t hw,
                                 const prec_t gravity, const prec_t n, const prec_t kappa) {
	return -gravity * n*n * sqrtf(hz*hz + hw*hw) *
	       powf(2.0f*h / (h*h + fmaxf(h*h, kappa*kappa)), 7.0f/3.0f) * hz;
}

HOST DEVICE
inline prec_t computeH(const prec_t Up, const prec_t Fp, const prec_t Um,
                      const prec_t Fm, const prec_t cm, const prec_t cp) {
	return ((cp*Fm-cm*Fp) + (cp*cm)*(Up-Um)) / (cp-cm);
}

HOST DEVICE
inline void computeInterface(const prec_t  w,     const prec_t  hu,     const prec_t  hv,
                             const prec_t  wp1,   const prec_t  hup1,   const prec_t  hvp1,
                             const prec_t  sw,    const prec_t  shu,    const prec_t  shv,
                             const prec_t  swp1,  const prec_t  shup1,  const prec_t  shvp1,
                             const prec_t  B,     const prec_t  Bp1,    const prec_t  Bp2,
                             const prec_t  B_ij,  const prec_t  Bp1_ij, const prec_t  kappa,
                                   prec_t* wp,          prec_t* hup,          prec_t* hvp,
                                   prec_t* wm,          prec_t* hum,          prec_t* hvm) {
	// Compute the interfacial water surface elevation values of the center cell.
	prec_t wcp = w + sw;
	prec_t wcm = w - sw;

	// Compute the interfacial water surface values of the east/north cell.
	prec_t wp1p = wp1 + swp1;
	prec_t wp1m = wp1 - swp1;

	if (wcp < Bp1 || wcm < B) {
		wcp = (w - B_ij) + Bp1;
		wcm = (w - B_ij) + B;
	}

	if (wp1p < Bp2 || wp1m < Bp1) {
		wp1p = (wp1 - Bp1_ij) + Bp2;
		wp1m = (wp1 - Bp1_ij) + Bp1;
	}

	*wm = wcp;
	*wp = wp1m;

	*hum = (wcp  - Bp1) * (computeVelocity(w   - B_ij,   hu,   kappa) + shu);
	*hup = (wp1m - Bp1) * (computeVelocity(wp1 - Bp1_ij, hup1, kappa) - shup1);
	*hvm = (wcp  - Bp1) * (computeVelocity(w   - B_ij,   hv,   kappa) + shv);
	*hvp = (wp1m - Bp1) * (computeVelocity(wp1 - Bp1_ij, hvp1, kappa) - shvp1);
}

HOST DEVICE
inline void computeF(const prec_t w, const prec_t B, const prec_t gravity, const prec_t kappa,
                     const prec_t epsilon, prec_t* hu, prec_t* hv, prec_t* ap,
                     prec_t* am, prec_t* F0, prec_t* F1, prec_t* F2) {
	if (w > B + epsilon) {
		prec_t h = w - B;
		prec_t u = *hu / h;

		*ap = u + sqrtf(gravity * h);
		*am = u - sqrtf(gravity * h);

		*F0 = (*hu);
		*F1 = (*hu) * (*hu) / h + 0.5f*gravity * h*h;
		*F2 = (*hu) * (*hv) / h;
	} else {
		*hu = 0.0f;
		*hv = 0.0f;
		*F0 = 0.0f;
		*F1 = 0.0f;
		*F2 = 0.0f;
		*ap = 0.0f;
		*am = 0.0f;
	}
}

HOST DEVICE
inline void computeG(const prec_t w, const prec_t B, const prec_t gravity, const prec_t kappa,
                     const prec_t epsilon, prec_t* hu, prec_t* hv, prec_t* bp,
                     prec_t* bm, prec_t* G0, prec_t* G1, prec_t* G2) {
	if (w > B + epsilon) {
		prec_t h = w - B;
		prec_t v = *hv / h;

		*bp = v + sqrtf(gravity * h);
		*bm = v - sqrtf(gravity * h);

		*G0 = (*hv);
		*G1 = (*hu) * (*hv) / h;
		*G2 = (*hv) * (*hv) / h + 0.5f*gravity * h*h;
	} else {
		*hu = 0.0f;
		*hv = 0.0f;
		*G0 = 0.0f;
		*G1 = 0.0f;
		*G2 = 0.0f;
		*bp = 0.0f;
		*bm = 0.0f;
	}
}
