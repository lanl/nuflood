#pragma once

#include <string>

class Folder {
public:
	Folder(void);
	Folder(const std::string& path);

	bool Exists(void) const;
	void Set(std::string path);
	void Clear(void);

	std::string path(void) const { return path_; }

private:
	std::string path_;
};
