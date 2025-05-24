#include "file.h"
#include "error.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

struct both_slashes {
    auto operator()(char first, char second) const -> bool {
        return first == '/' && second == '/';
    }
};

File::File() = default;

File::File(const std::string path) { Load(path); }

void File::Load(const std::string path) {
    ClearContent();
    path_ = path;

    if (!Exists()) {
        PrintErrorAndExit("File '" + path_ + "' does not exist.");
    } else {
        if (path_.at(0) != '/') {
            auto tmp_path = std::unique_ptr<char, decltype(&std::free)>(
                realpath(path_.c_str(), nullptr), &std::free);
            if (tmp_path) {
                path_ = std::string(tmp_path.get());
            }
        }

        path_.erase(std::unique(path_.begin(), path_.end(), both_slashes()),
                    path_.end());

        std::ifstream input_file(path_);
        std::stringstream buffer;
        buffer << input_file.rdbuf();
        content_ = buffer.str();
    }
}

void File::Clear() {
    ClearPath();
    ClearContent();
}

void File::ClearPath() { path_.clear(); }

void File::ClearContent() { content_.clear(); }

auto File::Exists() const -> bool {
    std::ifstream file_stream(path_);
    if (file_stream.good()) {
        file_stream.close();
        return true;
    }
    file_stream.close();
    return false;
}

auto File::IsEmpty() const -> bool { return content_.empty(); }
