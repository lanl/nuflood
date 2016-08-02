#include <limits>
#include <common/parameter.h>
#include "time.h"

Time::Time(const rapidjson::Value& root, const Grid<prec_t>& interpolated_grid) : ITime(root) {
	max_step_.Copy(interpolated_grid);
	max_step_.Fill((prec_t)0);
}
