#include "document.h"
#include "error.h"
#include "file.h"
#include <array>
#include <memory>

Document::Document(const File &file) {
    std::unique_ptr<FILE, decltype(&fclose)> p_file(
        fopen(file.path().c_str(), "r"), &fclose);

    if (!p_file) {
        PrintErrorAndExit("Failed to open file '" + file.path() + "'.");
    }

    constexpr size_t BUFFER_SIZE = 65536;
    std::array<char, BUFFER_SIZE> buffer{};
    rapidjson::FileReadStream file_read_stream(p_file.get(), buffer.data(),
                                               buffer.size());
    root.ParseStream<0>(file_read_stream);

    if (!root.IsObject()) {
        PrintErrorAndExit("File '" + file.path() +
                          "' is not a valid JSON document.");
    }
}
