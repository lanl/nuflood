#pragma once

#include <string>

class File {
public:
	File(const std::string path);
	File(void);

	void Load(const std::string path);
	void Clear(void);
	void ClearPath(void);
	void ClearContent(void);
	bool Exists(void) const;
	bool IsEmpty(void) const;

	std::string path() const { return path_; }
	std::string folder_path() const { return folder_path_; }
	std::string content() const { return content_; }
	void set_path(const std::string path) { path_ = path; }

private:
	std::string path_;
	std::string folder_path_;
	std::string content_;
};
