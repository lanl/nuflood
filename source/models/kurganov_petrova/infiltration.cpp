#include <common/parameter.h>
#include "infiltration.h"

Infiltration::Infiltration(const rapidjson::Value& root, const Topography& topography) {
	K_value_ = (prec_t)0; // Units are m/s.
	psi_value_ = (prec_t)0; // Units are m.
	dtheta_value_ = (prec_t)0; // Unitless (volume/volume).

	if (root.HasMember("infiltration")) {
		const rapidjson::Value& infiltration_json = root["infiltration"];
		ReadParameter(infiltration_json, "hydraulicConductivityFile", K_file_);
		ReadParameter(infiltration_json, "hydraulicConductivityValue", K_value_);
		ReadParameter(infiltration_json, "suctionHeadFile", psi_file_);
		ReadParameter(infiltration_json, "suctionHeadValue", psi_value_);
		ReadParameter(infiltration_json, "moistureDeficitFile", dtheta_file_);
		ReadParameter(infiltration_json, "moistureDeficitValue", dtheta_value_);
	}

	if (!K_file_.IsEmpty()) {
		// Assumes units of m/s.
		K_grid_.Load(K_file_);
		K_grid_.BilinearInterpolate();
		K_grid_.AddBoundaries();
		K_grid_.set_name("hydraulicConductivity");
		K_value_ = (prec_t)0; // If a hydraulic conductivity grid has been defined, reset K_value_ to zero.
	}

	if (!psi_file_.IsEmpty()) {
		psi_grid_.Load(psi_file_);
		psi_grid_.BilinearInterpolate();
		psi_grid_.AddBoundaries();
		psi_grid_.set_name("suctionHead");
		psi_value_ = (prec_t)0; // If a suction head grid has been defined, reset psi_value_ to zero.
	}

	if (!dtheta_file_.IsEmpty()) {
		dtheta_grid_.Load(dtheta_file_);
		dtheta_grid_.BilinearInterpolate();
		dtheta_grid_.AddBoundaries();
		dtheta_grid_.set_name("moistureDeficit");
		dtheta_value_ = (prec_t)0; // If a moisture deficit grid has been defined, reset dtheta_value_ to zero.
	}

	if (!K_file_.IsEmpty() || K_value_ > (prec_t)0) {
		F_.Copy(topography.elevation_interpolated());
		F_.Fill((prec_t)0);
		F_old_.Copy(F_);
		dF_.Copy(F_);
	}
}
