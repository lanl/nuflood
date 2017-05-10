#include <sstream>
#include <string>
#include <fstream>
#include "error.h"
#include "folder.h"

Folder::Folder(void) {
	path_ = "";
}

Folder::Folder(const std::string& path) {
	Set(path);
}

void Folder::Set(std::string path) {
	path_ = path;

	if (!path_.empty() && *path_.rbegin() != '/') {
		path_ = path_ + '/';
	}

	if (!Exists()) {
		PrintErrorAndExit("Folder '" + path_ + "' does not exist.");
	}
}

void Folder::Clear(void) {
	path_ = "";
}

bool Folder::Exists(void) const {
	std::ifstream f(path_);
	if (f.good()) {
		f.close();
		return true;
	} else {
		f.close();
		return false;
	}
}
