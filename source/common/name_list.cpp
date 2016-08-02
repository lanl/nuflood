#include <algorithm>
#include <iterator>
#include <iostream>
#include "name_list.h"

void NameList::Add(const std::string name) {
	if (!Contains(name)) {
		name_list_.push_back(name);
	}
}

bool NameList::Contains(const std::string name) const {
	if (std::find(name_list_.begin(), name_list_.end(), name) !=
	    name_list_.end()) {
		return true;
	} else {
		return false;
	}
}

void NameList::Clear(void) {
	name_list_.clear();
}

void NameList::PrintContents(void) const {
	std::vector<std::string>::const_iterator it;
	for (it = name_list_.begin(); it != name_list_.end(); ++it) {
		std::cout << *it << std::endl;
	}
}
