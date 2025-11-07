#include "handler.hh"

#include <regex>
#include <string>

namespace HTTP {
  std::string JsonHandler::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    size_t last = str.find_last_not_of(" \t\r\n");
    return (first == std::string::npos || last == std::string::npos)
      ? ""
      : str.substr(first, last - first + 1);
  }
  // Function to parse the JSON-like string into a std::map
  void JsonHandler::parse_json(const std::string& json_string, std::map<std::string, std::string>& result) {
    // Key is a string, value is either a string or any non-comma/non-closing-brace text
    std::regex re(R"rs("([^"]+)"\s*:\s*("[^"]*"|[^,}]*))rs");
    std::smatch match;
    std::string input = json_string;

    // Remove outer curly braces if present
    if (input.front() == '{' && input.back() == '}') {
        input = input.substr(1, input.length() - 2);  // Strip off outer {}
    }

    // Search for key-value pairs
    while (std::regex_search(input, match, re)) {
        std::string key = match[1].str();    // The key is captured in the first group
        std::string value = match[2].str();  // The value is captured in the second group

        // Clean the key and value by trimming spaces
        key = trim(key);
        value = trim(value);

        // Remove quotes around values if they exist
        if (value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }

        // Insert into the map
        result[key] = value;

        // Move to the next part of the string after the matched key-value pair
        input = match.suffix().str();
    }
  }
}
