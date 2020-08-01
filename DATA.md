# Time	
- `start` and `end` times are defined with respect to other temporal data.
- These data are in units of seconds.
- If these data are not provided, the simulation’s start time is assumed to be zero, and the end time is assumed to be the maximum allowable floating point value.
- `timeStep` is a maximum time step that may be defined by the user. This will override the time step suggested by the [CFL condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) when applicable.
- `maxIterations` is an optional user-defined maximum number of iterations for which the simulation will execute before termination.

# Digital elevation model for topographic/bathymetric elevations.
- `elevationFile` is a path to a raster in ARC/INFO ASCII grid format. The raster is assumed to use units of meters above some datum in the vertical dimension.
- If a digital elevation model is not provided, a flat grid with 256x256 cells and a one meter grid cell size is instead initialized for the scenario.
- `metric` is a Boolean variable defining if a grid uses UTM (i.e., metric) or decimal degree reference coordinates and cell sizes in the header metadata.

# Conserved variables
- Initial `waterSurfaceElevation`, `horizontalDischarge`, `verticalDischarge`, and/or `depth` values may be defined using ARC/INFO ASCII grids or float constants.
- If undefined, `waterSurfaceElevation` assumes a grid equivalent to the digital elevation model. Also if undefined, `horizontalDischarge`, `verticalDischarge`, and `depth` grids are assumed to be filled with zeroes.

# Bed friction	
- `manningFile` is a raster in ARC/INFO ASCII grid format.
  - Typically, ARC/INFO ASCII grids of roughness are calibrated and defined with respect to [National Land Cover Database (NLCD)](https://www.mrlc.gov/) land type data.
- `manningValue` is a single global roughness coefficient used over the domain.
- These data must be defined in units of Manning’s roughness coefficient.
- If frictional data are not defined, the bed surface is assumed to be frictionless.

# Point sources	
- Some flooding phenomena may be described using one or multiple point sources from which water volume is temporally discharged.
- This data is formatted using a comma-separated time series and a point location.
  - A comma-separated file may include a heading starting with a `#` symbol.
  - Units should be in the format of `time (s), rate (cubic meters per second).`
  - The point location must be defined using decimal degrees (if `metric` is defined as false) or metric (UTM) coordinates (if `metric` is defined as true).	
- Multiple point sources may be defined, but the number of points is typically small.

# Rainfall	
- `grid` is a path to an ARC/INFO ASCII raster that spatially defines the amount of precipitation that falls over a given area (in meters) during the simulation.	
- `stormCurve` is a path to a CSV file that scales this amount of water temporally (e.g., an [SCS or "design" storm curve](http://www.ce.utexas.edu/prof/maidment/GradHydro2010/Visual/DesignStorms.ppt)) in an isotropic manner.
- If rainfall is not defined, a rainfall rate of zero is assumed.

# Soil parameters
- The Green-Ampt infiltration model is used.
  - `suctionHead` is the wetting front soil suction head.
  - `soilMoistureDeficit` is the soil moisture deficit.
  - `hydraulicConductivity` is the saturated hydraulic conductivity.
- `suctionHead`, `soilMoistureDeficit`, and `hydraulicConductivity` must each always be defined when modeling infiltration.
  - Each of these parameters may be defined using either an ARC/INFO ASCII grid or a single-valued float constant.
  - `suctionHead` is defined in units of meters; `soilMoistureDeficit` is unitless; and `hydraulicConductivity` is defined in units of meters per second.
- If `suctionHead`, `soilMoistureDeficit`, or `hydraulicConductivity` is undefined, infiltration is not modeled.

# Boundary conditions
- `east`, `west`, `north`, and `south` boundaries may each be defined.
- Boundary conditions include `none`, `open`, `wall`, `criticalDepth`, or `marigram`.
  - If undefined, boundary conditions are assumed as `none`.
  - `marigram` boundary conditions are typically used for simulating tsunamis.
    - The marigram is formatted as a comma-separated time series.
    - The comma-separated file may include a heading commented with a `#` symbol.	
    - The comma-separated file should be with units of `time (s), water surface elevation (m)`.
      - Care should be taken to ensure the water surface elevation never reaches a value less than the corresponding digital elevation model value along the boundary.
    - This marigram file should be specified as a `marigram` in the `sources` parameter category.

# Output options	
- The output time step should be in units of seconds.
- The output folder should be an existing directory on the user’s machine (and include a trailing `/`).	
- The following ARC/INFO ASCII grids may be outputted temporally: `depth`, `waterSurfaceElevation`, `horizontalDischarge`, `verticalDischarge`, `unitDischarge`, `maxDepth`, and `maxUnitDischarge`.
- The following ARC/INFO ASCII grids may be outputted statically (but updated after each output timestep): `maxDepth`, `maxUnitDischarge`.
- To keep track of volume conservation error, the following may be toggled (via Boolean variables) to output via command line: `printTime`, 	`printTimeStep`, `printIteration`, `printVolumeComputed`, `printVolumeAdded`, `printVolumeConservationError`.
  - These values will be printed during each output time step.

# Constants		
- Gravitational acceleration may be defined; if not, 9.80665 m/s^2 is assumed.
- The velocity desingularization constant may be defined; smaller values indicate the depths at which velocities are corrected using the desingularization function. If not, a value of `sqrt(0.01*max(max(1.0, dx), dy))` is assumed.
- The machine epsilon may be redefined. This value controls the minimum depth, below which depths are numerically truncated to be zero.
- The use of the wet cell tracking algorithm is defined as a Boolean variable.
  - This should be `true` when simulating sparse events (e.g., point sources) and `false` when simulating wide area inundation (e.g., extreme rainfall).
