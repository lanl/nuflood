#pragma once

#include <vector>

class NameList {
public:
	void Add(const std::string name);
	bool Contains(const std::string name) const;
	void Clear(void);
	void PrintContents(void) const;

private:
	std::vector<std::string> name_list_;
};
