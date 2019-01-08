#pragma once

#include <iostream>
#include <vector>
#include <rapidjson/filereadstream.h>
#include <rapidjson/document.h>
#include <rapidjson/schema.h>
#include <rapidjson/stringbuffer.h>
#include "point_source.h"
#include "schema.h"

//! Class that stores input arguments for cellular automata model execution.
class CellularAutomataInput {
public:
	// Constructor.
	CellularAutomataInput(std::string path);

	//! Returns the path to the topographic elevation Raster.
	/*! \return Path to the topographic elevation Raster. */
	const std::string elevation_path(void) const { return elevation_path_; }

	//! Returns the path to the initial depth Raster.
	/*! \return Path to the initial depth Raster. */
	const std::string depth_path(void) const { return depth_path_; }

	//! Returns the path to the initial water surface elevation Raster.
	/*! \return Path to the initial water surface elevation Raster. */
	const std::string wse_path(void) const { return wse_path_; }

	//! Returns the path to the output depth Raster.
	/*! \return Path to the output depth Raster. */
	const std::string output_depth_path(void) const { return output_depth_path_; }

	//! Returns the path to the output water surface elevation Raster.
	/*! \return Path to the output water surface elevation Raster. */
	const std::string output_wse_path(void) const { return output_wse_path_; }

	//! Returns the path to the output summary JSON.
	/*! \return Path to the output summary JSON. */
	const std::string output_summary_path(void) const { return output_summary_path_; }

	//! Returns a reference to the vector of depth point sources.
	/*! \return A reference to the vector of depth point sources. */
	const std::vector<PointSource>& point_sources_depth(void) const { return point_sources_depth_; }

	//! Returns a reference to the vector of water surface elevation point sources.
	/*! \return A reference to the vector of water surface elevation point sources. */
	const std::vector<PointSource>& point_sources_wse(void) const { return point_sources_wse_; }

protected:
	std::string elevation_path_;
	std::string depth_path_;
	std::string wse_path_; // wse := water surface elevation
	std::string output_depth_path_;
	std::string output_wse_path_;
	std::string output_summary_path_;
	std::vector<PointSource> point_sources_depth_;
	std::vector<PointSource> point_sources_wse_;
};
