#pragma once

#include "error.h"
#include "file.h"
#include "folder.h"
#include "name_list.h"
#include "document.h"
#include "point_source_list.h"

inline void ReadParameter(const rapidjson::Value& root,
                          const std::string parameter_category, File& file) {
	if (root.HasMember(parameter_category.c_str())) {
		if (root[parameter_category.c_str()].IsString()) {
			file.Load(root[parameter_category.c_str()].GetString());
		} else {
			PrintErrorAndExit("File path '" + parameter_category +
			                  "' is not a valid string.");
		}
	} else {
		file.Clear();
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, double& value) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsDouble()) {
			value = root[name.c_str()].GetDouble();
		} else {
			PrintErrorAndExit("Value '" + name + "' is not a valid double.");
		}
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, float& value) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsDouble()) {
			value = (float)(root[name.c_str()].GetDouble());
		} else {
			PrintErrorAndExit("Value '" + name + "' is not a valid float.");
		}
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, unsigned int& value) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsUint()) {
			value = (unsigned int)(root[name.c_str()].GetUint());
		} else {
			PrintErrorAndExit("Value '" + name + "' is not a valid unsigned integer.");
		}
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, unsigned long& value) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsUint()) {
			value = (unsigned long)(root[name.c_str()].GetUint());
		} else {
			PrintErrorAndExit("Value '" + name + "' is not a valid unsigned integer.");
		}
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, bool& value) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsBool()) {
			value = root[name.c_str()].GetBool();
		} else {
			PrintErrorAndExit("Value '" + name + "' is not a valid boolean.");
		}
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, Folder& folder) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsString()) {
			folder.Set(root[name.c_str()].GetString());
		} else {
			PrintErrorAndExit("Folder path '" + name + "' is not a valid string.");
		}
	} else {
		folder.Clear();
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string array_name, NameList& name_list) {	
	if (root.HasMember(array_name.c_str())) {
		if (root[array_name.c_str()].IsArray()) {
			const rapidjson::Value& name_array_json = root[array_name.c_str()];

			for (rapidjson::SizeType i = 0; i < name_array_json.Size(); i++) { 
				if (name_array_json[i].IsString()) {
					std::string name = name_array_json[i].GetString();
					name_list.Add(name);
				} else {
					PrintErrorAndExit("Array '" + array_name + "' contains an invalid string.");
				}
			}
		} else {
			PrintErrorAndExit("Array '" + array_name + "' is not a valid array.");
		}
	} else {
		name_list.Clear();
	}
}

inline void ReadParameter(const rapidjson::Value& root, const std::string name, std::string& value) {
	if (root.HasMember(name.c_str())) {
		if (root[name.c_str()].IsString()) {
			value = root[name.c_str()].GetString();
		} else {
			PrintErrorAndExit("Parameter '" + name + "' is not a valid string.");
		}
	} else {
		value = "";
	}
}

template<class T>
inline void ReadParameter(const rapidjson::Value& root, const std::string array_name, PointSourceList<T>& point_source_list) {	
	if (root.HasMember(array_name.c_str())) {
		if (root[array_name.c_str()].IsArray()) {
			const rapidjson::Value& point_array_json = root[array_name.c_str()];

			unsigned int i = 0;
			while (i < point_array_json.Size()) {
				double x = 0.0, y = 0.0;
				if (point_array_json[i].IsDouble()) {
					x = point_array_json[i].GetDouble();
					i += 1;
				} else {
					PrintErrorAndExit("Point source x-coordinate is not a valid double.");
				}

				if (point_array_json[i].IsDouble()) {
					y = point_array_json[i].GetDouble();
					i += 1;
				} else {
					PrintErrorAndExit("Point source y-coordinate is not a valid double.");
				}

				File file;
				if (point_array_json[i].IsString()) {
					std::string file_path = point_array_json[i].GetString();
					file.Load(file_path);
					i += 1;
				} else {
					PrintErrorAndExit("Point source file path is not a valid string.");
				}

				point_source_list.Add(PointSource<T>(x, y, file));
			}
		} else {
			PrintErrorAndExit("Array '" + array_name + "' is not a valid array.");
		}
	}
}
