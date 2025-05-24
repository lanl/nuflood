#pragma once

#include <string>

class File {
  public:
    File(std::string path);
    File();

    void Load(std::string path);
    void Clear();
    void ClearPath();
    void ClearContent();
    auto Exists() const -> bool;
    auto IsEmpty() const -> bool;

    auto path() const -> std::string { return path_; }
    auto folder_path() const -> std::string { return folder_path_; }
    auto content() const -> std::string { return content_; }
    void set_path(const std::string path) { path_ = path; }

  private:
    std::string path_;
    std::string folder_path_;
    std::string content_;
};
