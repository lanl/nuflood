#include <math.h>
#include "update_boundaries.h"

void UpdateBoundaries(const BoundaryConditions& BC, const Constants& C, const Topography& B, const Time& T, Conserved& U, Sources& R) {
	switch (BC.types.east) {
		case NONE:
			break;
		case OPEN:
			UpdateEastBoundaryOpen(C, U);
			break;
		case WALL:
			UpdateEastBoundaryWall(C, B, U);
			break;
		case CRITICAL_DEPTH:
			UpdateEastBoundaryCriticalDepth(C, B, U);
			break;
		case MARIGRAM:
			UpdateEastBoundaryMarigram(C, B, R, T, U);
			break;
	}

	switch (BC.types.west) {
		case NONE:
			break;
		case OPEN:
			UpdateWestBoundaryOpen(C, U);
			break;
		case WALL:
			UpdateWestBoundaryWall(C, B, U);
			break;
		case CRITICAL_DEPTH:
			UpdateWestBoundaryCriticalDepth(C, B, U);
			break;
		case MARIGRAM:
			UpdateWestBoundaryMarigram(C, B, R, T, U);
			break;
	}

	switch (BC.types.north) {
		case NONE:
			break;
		case OPEN:
			UpdateNorthBoundaryOpen(C, U);
			break;
		case WALL:
			UpdateNorthBoundaryWall(C, B, U);
			break;
		case CRITICAL_DEPTH:
			UpdateNorthBoundaryCriticalDepth(C, B, U);
			break;
		case MARIGRAM:
			UpdateNorthBoundaryMarigram(C, B, R, T, U);
			break;
	}

	switch (BC.types.south) {
		case NONE:
			break;
		case OPEN:
			UpdateSouthBoundaryOpen(C, U);
			break;
		case WALL:
			UpdateSouthBoundaryWall(C, B, U);
			break;
		case CRITICAL_DEPTH:
			UpdateSouthBoundaryCriticalDepth(C, B, U);
			break;
		case MARIGRAM:
			UpdateSouthBoundaryMarigram(C, B, R, T, U);
			break;
	}
}

inline void UpdateEastBoundaryOpen(const Constants& C, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(j) shared(num_columns, num_rows, hu, hv)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE east    = j*num_columns + (num_columns-1);
		INT_TYPE east_m1 = j*num_columns + (num_columns-2);
		INT_TYPE east_m2 = j*num_columns + (num_columns-3);

		 w[east] =  w[east_m2];
		hu[east] = hu[east_m2];
		hv[east] = hv[east_m2];

		 w[east_m1] =  w[east_m2];
		hu[east_m1] = hu[east_m2];
		hv[east_m1] = hv[east_m2];
	}
}

inline void UpdateEastBoundaryWall(const Constants& C, const Topography& B, Conserved& U) {
	//static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	//static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(j) shared(num_columns, num_rows, hu, hv)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE east    = j*num_columns + (num_columns-1);
		INT_TYPE east_m1 = j*num_columns + (num_columns-2);
		INT_TYPE east_m2 = j*num_columns + (num_columns-3);
		INT_TYPE east_m3 = j*num_columns + (num_columns-4);

		// w[east] =   w[east_m2]; //Bi[east] + h_east_m3;
		// w[east] =  Bi[east_m2];
		hu[east] = -hu[east_m2];
		hv[east] =  hv[east_m2];

		// w[east_m1] =   w[east_m3];
		// w[east_m1] =  Bi[east_m3];
		hu[east_m1] = -hu[east_m3];
		hv[east_m1] =  hv[east_m3];
	}
}

inline void UpdateEastBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(j) shared(num_columns, num_rows, w, hu, hv, Bi)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE east    = j*num_columns + (num_columns-1);
		INT_TYPE east_m1 = j*num_columns + (num_columns-2);
		INT_TYPE east_m2 = j*num_columns + (num_columns-3);

		prec_t w_new;
		prec_t h = w[east_m2] - Bi[east_m2];
		if (fabsf(hu[east_m2]) > (prec_t)0 && h > (prec_t)0) {
			// The Froude number, F, can be defined by u / sqrt(g*h). Critical depth
			// occurs at a Froude number of unity. Thus, for a critical depth
			// boundary condition, 1 = u^2 / (g*h). We simply need to set h with
			// respect to v, i.e., h = u^2 / g, or h = (hu)*(hu) / (g * h*h).
			prec_t h_c = (hu[east_m2]*hu[east_m2]) / (C.gravitational_acceleration() * h*h);
			h_c = fminf(h, h_c);
			w_new = Bi[east_m2] + h_c;
		} else {
			w_new = w[east_m2];
		}

		w [east] = w_new;
		hu[east] = hu[east_m2];
		hv[east] = hv[east_m2];

		w [east_m1] = w_new;
		hu[east_m1] = hu[east_m2];
		hv[east_m1] = hv[east_m2];
	}
}

inline void UpdateEastBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	prec_t marigram_elevation = R.marigram().interpolated_value(T.current());

	#pragma omp parallel for private(j) shared(num_columns, num_rows, w, hu, hv)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE east    = j*num_columns + (num_columns-1);
		INT_TYPE east_m1 = j*num_columns + (num_columns-2);

		w [east] = marigram_elevation;
		hu[east] = (prec_t)0;
		hv[east] = (prec_t)0;

		w [east_m1] = marigram_elevation;
		hu[east_m1] = (prec_t)0;
		hv[east_m1] = (prec_t)0;
	}
}

inline void UpdateWestBoundaryOpen(const Constants& C, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(j) shared(num_columns, num_rows, hu, hv)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE west    = j*num_columns + 0;
		INT_TYPE west_p1 = j*num_columns + 1;
		INT_TYPE west_p2 = j*num_columns + 2;

		 w[west] =  w[west_p2];
		hu[west] = hu[west_p2];
		hv[west] = hv[west_p2];

		 w[west_p1] =  w[west_p2];
		hu[west_p1] = hu[west_p2];
		hv[west_p1] = hv[west_p2];
	}
}

inline void UpdateWestBoundaryWall(const Constants& C, const Topography& B, Conserved& U) {
	//static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	//static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(j) shared(num_columns, num_rows, hu, hv)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE west    = j*num_columns + 0;
		INT_TYPE west_p1 = j*num_columns + 1;
		INT_TYPE west_p2 = j*num_columns + 2;
		INT_TYPE west_p3 = j*num_columns + 3;

		//prec_t h_west    = w[west   ] - Bi[west   ];
		//prec_t h_west_p1 = w[west_p1] - Bi[west_p1];
		//prec_t h_west_p2 = w[west_p2] - Bi[west_p2];
		//prec_t h_west_p3 = w[west_p3] - Bi[west_p3];

		//w [west] =  Bi[west] + h_west_p3;
		hu[west] = -hu[west_p2];
		hv[west] =  hv[west_p2];

		//w [west_p1] =  Bi[west_p1] + h_west_p2;
		hu[west_p1] = -hu[west_p3];
		hv[west_p1] =  hv[west_p3];
	}
}

inline void UpdateWestBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(j) shared(num_columns, num_rows, w, hu, hv, Bi)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE west    = j*num_columns + 0;
		INT_TYPE west_p1 = j*num_columns + 1;
		INT_TYPE west_p2 = j*num_columns + 2;

		prec_t w_new;
		prec_t h = w[west_p2] - Bi[west_p2];
		if (fabsf(hu[west_p2]) > (prec_t)0 && h > (prec_t)0) {
			// The Froude number, F, can be defined by u / sqrt(g*h). Critical
			// depth occurs at a Froude number of unity. Thus, for a critical
			// depth boundary condition, 1 = u^2 / (g*h). We simply need to set h
			// with respect to u, i.e., h = u^2 / g, or h = (hu)*(hu) / (g * h*h).
			prec_t h_c = (hu[west_p2]*hu[west_p2]) / (C.gravitational_acceleration() * h*h);
			h_c = fminf(h, h_c);
			w_new = Bi[west_p2] + h_c;
		} else {
			w_new = w[west_p2];
		}

		w [west] = w_new;
		hu[west] = hu[west_p2];
		hv[west] = hv[west_p2];

		w [west_p1] = w_new;
		hu[west_p1] = hu[west_p2];
		hv[west_p1] = hv[west_p2];
	}
}

inline void UpdateWestBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE j;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	prec_t marigram_elevation = R.marigram().interpolated_value(T.current());

	#pragma omp parallel for private(j) shared(num_columns, num_rows, w, hu, hv)
	for (j = 2; j < num_rows - 2; j++) {
		INT_TYPE west    = j*num_columns + 0;
		INT_TYPE west_p1 = j*num_columns + 1;

		w [west] = marigram_elevation;
		hu[west] = (prec_t)0;
		hv[west] = (prec_t)0;

		w [west_p1] = marigram_elevation;
		hu[west_p1] = (prec_t)0;
		hv[west_p1] = (prec_t)0;
	}
}

inline void UpdateNorthBoundaryOpen(const Constants& C, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(i) shared(num_columns, num_rows, hu, hv)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE north    = (num_rows-1)*num_columns + i;
		INT_TYPE north_m1 = (num_rows-2)*num_columns + i;
		INT_TYPE north_m2 = (num_rows-3)*num_columns + i;

		 w[north] =  w[north_m2];
		hu[north] = hu[north_m2];
		hv[north] = hv[north_m2];

		 w[north_m1] =  w[north_m2];
		hu[north_m1] = hu[north_m2];
		hv[north_m1] = hv[north_m2];
	}
}

inline void UpdateNorthBoundaryWall(const Constants& C, const Topography& B, Conserved& U) {
	//static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	//static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(i) shared(num_columns, num_rows, hu, hv)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE north    = (num_rows-1)*num_columns + i;
		INT_TYPE north_m1 = (num_rows-2)*num_columns + i;
		INT_TYPE north_m2 = (num_rows-3)*num_columns + i;
		INT_TYPE north_m3 = (num_rows-4)*num_columns + i;

		//prec_t h_north    = w[north   ] - Bi[north   ];
		//prec_t h_north_m1 = w[north_m1] - Bi[north_m1];
		//prec_t h_north_m2 = w[north_m2] - Bi[north_m2];
		//prec_t h_north_m3 = w[north_m3] - Bi[north_m3];
	
		//w [north] =  Bi[north] + h_north_m3;
		hu[north] =  hu[north_m2];
		hv[north] = -hv[north_m2];

		//w [north_m1] =  Bi[north_m1] + h_north_m2;
		hu[north_m1] =  hu[north_m3];
		hv[north_m1] = -hv[north_m3];
	}
}

inline void UpdateNorthBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	#pragma omp parallel for private(i) shared(num_columns, num_rows, w, hu, hv, Bi)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE north    = (num_rows-1)*num_columns + i;
		INT_TYPE north_m1 = (num_rows-2)*num_columns + i;
		INT_TYPE north_m2 = (num_rows-3)*num_columns + i;

		prec_t w_new;
		prec_t h = w[north_m2] - Bi[north_m2];
		if (fabsf(hv[north_m2]) > (prec_t)0 && h > (prec_t)0) {
			// The Froude number, F, can be defined by v / sqrt(g*h). Critical depth
			// occurs at a Froude number of unity. Thus, for a critical depth
			// boundary condition, 1 = v^2 / (g*h). We simply need to set h with
			// respect to v, i.e., h = v^2 / g, or h = (hv)*(hv) / (g * h*h).
			prec_t h_c = (hv[north_m2]*hv[north_m2]) / (C.gravitational_acceleration() * h*h);
			h_c = fminf(h, h_c);
			w_new = Bi[north_m2] + h_c;
		} else {
			w_new = w[north_m2];
		}

		w [north] = w_new;
		hu[north] = hu[north_m2];
		hv[north] = hv[north_m2];

		w [north_m1] = w_new;
		hu[north_m1] = hu[north_m2];
		hv[north_m1] = hv[north_m2];
	}
}

inline void UpdateNorthBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();
	INT_TYPE num_rows    = C.num_rows();

	prec_t marigram_elevation = R.marigram().interpolated_value(T.current());

	#pragma omp parallel for private(i) shared(num_columns, num_rows, w, hu, hv)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE north    = (num_rows-1)*num_columns + i;
		INT_TYPE north_m1 = (num_rows-2)*num_columns + i;

		w [north] = marigram_elevation;
		hu[north] = (prec_t)0;
		hv[north] = (prec_t)0;

		w [north_m1] = marigram_elevation;
		hu[north_m1] = (prec_t)0;
		hv[north_m1] = (prec_t)0;
	}
}

inline void UpdateSouthBoundaryOpen(const Constants& C, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();

	#pragma omp parallel for private(i) shared(num_columns, hu, hv)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE south    = 0*num_columns + i;
		INT_TYPE south_p1 = 1*num_columns + i;
		INT_TYPE south_p2 = 2*num_columns + i;

		 w[south] =  w[south_p2];
		hu[south] = hu[south_p2];
		hv[south] = hv[south_p2];

		 w[south_p1] =  w[south_p2];
		hu[south_p1] = hu[south_p2];
		hv[south_p1] = hv[south_p2];
	}
}

inline void UpdateSouthBoundaryWall(const Constants& C, const Topography& B, Conserved& U) {
	//static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	//static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();

	#pragma omp parallel for private(i) shared(num_columns, hu, hv)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE south    = 0*num_columns + i;
		INT_TYPE south_p1 = 1*num_columns + i;
		INT_TYPE south_p2 = 2*num_columns + i;
		INT_TYPE south_p3 = 3*num_columns + i;

		//prec_t h_south    = w[south   ] - Bi[south   ];
		//prec_t h_south_p1 = w[south_p1] - Bi[south_p1];
		//prec_t h_south_p2 = w[south_p2] - Bi[south_p2];
		//prec_t h_south_p3 = w[south_p3] - Bi[south_p3];
	
		//w [south] =  Bi[south] + h_south_p3;
		hu[south] =  hu[south_p2];
		hv[south] = -hv[south_p2];

		//w [south_p1] =  Bi[south_p1] + h_south_p2;
		hu[south_p1] =  hu[south_p3];
		hv[south_p1] = -hv[south_p3];
	}
}

inline void UpdateSouthBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();
	static prec_t* Bi = B.elevation_interpolated().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();

	#pragma omp parallel for private(i) shared(num_columns, w, hu, hv, Bi)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE south    = 0*num_columns + i;
		INT_TYPE south_p1 = 1*num_columns + i;
		INT_TYPE south_p2 = 2*num_columns + i;

		prec_t w_new;
		prec_t h = w[south_p2] - Bi[south_p2];
		if (fabsf(hv[south_p2]) > (prec_t)0 && h > (prec_t)0) {
			// The Froude number, F, can be defined by v / sqrt(g*h). Critical depth
			// occurs at a Froude number of unity. Thus, for a critical depth
			// boundary condition, 1 = v^2 / (g*h). We simply need to set h with
			// respect to v, i.e., h = v^2 / g, or h = (hv)*(hv) / (g * h*h).
			prec_t h_c = (hv[south_p2]*hv[south_p2]) / (C.gravitational_acceleration() * h*h);
			h_c = fminf(h, h_c);
			w_new = Bi[south_p2] + h_c;
		} else {
			w_new = w[south_p2];
		}

		w [south] = w_new;
		hu[south] = hu[south_p2];
		hv[south] = hv[south_p2];

		w [south_p1] = w_new;
		hu[south_p1] = hu[south_p2];
		hv[south_p1] = hv[south_p2];
	}
}

inline void UpdateSouthBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U) {
	static prec_t*  w = U. w().data();
	static prec_t* hu = U.hu().data();
	static prec_t* hv = U.hv().data();

	INT_TYPE i;
	INT_TYPE num_columns = C.num_columns();

	prec_t marigram_elevation = R.marigram().interpolated_value(T.current());

	#pragma omp parallel for private(i) shared(num_columns, w, hu, hv)
	for (i = 2; i < num_columns - 2; i++) {
		INT_TYPE south    = 0*num_columns + i;
		INT_TYPE south_p1 = 1*num_columns + i;

		w [south] = marigram_elevation;
		hu[south] = (prec_t)0;
		hv[south] = (prec_t)0;

		w [south_p1] = marigram_elevation;
		hu[south_p1] = (prec_t)0;
		hv[south_p1] = (prec_t)0;
	}
}
