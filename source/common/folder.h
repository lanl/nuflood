#pragma once

#include <string>

class Folder {
public:
	Folder();
	Folder(const std::string& path);

	auto Exists() const -> bool;
	void Set(std::string path);
	void Clear();

	auto path() const -> std::string { return path_; }

private:
	std::string path_;
};
