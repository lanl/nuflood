#pragma once

#include "constants.h"
#include "topography.h"
#include "conserved.h"
#include "sources.h"
#include "boundary_conditions.h"

class Constants;

void UpdateBoundaries(const BoundaryConditions& BC, const Constants& C, const Topography& B, const Time& T, Conserved& U, Sources& R);

inline void UpdateEastBoundaryOpen(const Constants& C, Conserved& U);
inline void UpdateEastBoundaryWall(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateEastBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateEastBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U);

inline void UpdateWestBoundaryOpen(const Constants& C, Conserved& U);
inline void UpdateWestBoundaryWall(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateWestBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateWestBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U);

inline void UpdateNorthBoundaryOpen(const Constants& C, Conserved& U);
inline void UpdateNorthBoundaryWall(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateNorthBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateNorthBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U);

inline void UpdateSouthBoundaryOpen(const Constants& C, Conserved& U);
inline void UpdateSouthBoundaryWall(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateSouthBoundaryCriticalDepth(const Constants& C, const Topography& B, Conserved& U);
inline void UpdateSouthBoundaryMarigram(const Constants& C, const Topography& B, const Sources& R, const Time& T, Conserved& U);
