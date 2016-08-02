#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h> 
#include <stdio.h>
#include <algorithm>
#include <string>
#include "error.h"
#include "file.h"

struct both_slashes {
	bool operator()(char a, char b) const {
		return a == '/' && b == '/';
	}
};

File::File(void) : path_(""), content_("") { }

File::File(const std::string path) {
	Load(path);
}

void File::Load(const std::string path) {
	ClearContent();
	path_ = path;

	if (!Exists()) {
		PrintErrorAndExit("File '" + path_ + "' does not exist.");
	} else {
		if (path_.at(0) != '/') {
			char* tmp_path = realpath(path_.c_str(), nullptr);
			path_ = std::string(tmp_path);
			free(tmp_path);
		}

		path_.erase(std::unique(path_.begin(), path_.end(), both_slashes()), path_.end());

		std::ifstream t(path_);
		std::stringstream buffer;
		buffer << t.rdbuf();
		content_ = buffer.str();
	}
}

void File::Clear(void) {
	ClearPath();
	ClearContent();
}

void File::ClearPath(void) {
	path_.clear();
}

void File::ClearContent(void) {
	content_.clear();
}

bool File::Exists(void) const {
	std::ifstream f(path_);
	if (f.good()) {
		f.close();
		return true;
	} else {
		f.close();
		return false;
	}
}

bool File::IsEmpty(void) const {
	return content_.empty();
}
