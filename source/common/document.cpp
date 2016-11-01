#include "document.h"
#include "error.h"
#include "file.h"

Document::Document(const File& file) {
	FILE* p_file = fopen(file.path().c_str(), "r");
	char buffer[65536];
	rapidjson::FileReadStream file_read_stream(p_file, buffer, sizeof(buffer));
	root.ParseStream<0>(file_read_stream);
	fclose(p_file);

	if (!root.IsObject()) {
		PrintErrorAndExit("File '" + file.path() + "' is not a valid JSON document.");
	}
}
