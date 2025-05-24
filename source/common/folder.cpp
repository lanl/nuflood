#include "folder.h"
#include "error.h"
#include <fstream>
#include <sstream>
#include <string>

Folder::Folder() = default;

Folder::Folder(const std::string &path) { Set(path); }

void Folder::Set(std::string path) {
    path_ = path;

    if (!path_.empty() && *path_.rbegin() != '/') {
        path_ = path_ + '/';
    }

    if (!Exists()) {
        PrintErrorAndExit("Folder '" + path_ + "' does not exist.");
    }
}

void Folder::Clear() { path_ = ""; }

auto Folder::Exists() const -> bool {
    std::ifstream file_stream(path_);
    if (file_stream.good()) {
        file_stream.close();
        return true;
    }
    file_stream.close();
    return false;
}
