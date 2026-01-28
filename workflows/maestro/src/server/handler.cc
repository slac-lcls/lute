#include "handler.hh"

#include <string>

namespace HTTP {
  std::string JsonHandler::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    size_t last = str.find_last_not_of(" \t\r\n");
    return (first == std::string::npos || last == std::string::npos)
      ? ""
      : str.substr(first, last - first + 1);
  }

  // Convert JSON strong to map
  // This is an attempt at an O(N) implementation - the previous version
  // was regex based and O(N^2) minimum
  void JsonHandler::parse_json(const std::string& json_string,
                               std::map<std::string, std::string>& result) {
    size_t i = 0;
    size_t len = json_string.length();

    // Skip leading whitespace and '{'
    while (i < len && (isspace(json_string[i]) || json_string[i] == '{')) {
      i++;
    }

    auto parse_string = [&](size_t& pos) -> std::string {
      std::string res;
      if (pos >= len || json_string[pos] != '"') {
        return "";
      }
      pos++; // skip "
      while (pos < len) {
        if (json_string[pos] == '\\' && pos + 1 < len) {
          if (json_string[pos + 1] == '"' || json_string[pos + 1] == '\\') {
            res += json_string[pos + 1];
            pos += 2;
            continue;
          }
        }
        if (json_string[pos] == '"') {
          pos++; // skip "
          return res;
        }
        res += json_string[pos++];
      }
      return res;
    };

    while (i < len) {
      // Skip whitespace, commas, and '}'
      while (i < len
             && (isspace(json_string[i])
                 || json_string[i] == ','
                 || json_string[i] == '}')) {
        i++;
      }
      if (i >= len) {
        break;
      }

      std::string key = parse_string(i);

      // Skip whitespace and ':'
      while (i < len && (isspace(json_string[i]) || json_string[i] == ':')) {
        i++;
      }
      if (i >= len) {
        break;
      }

      std::string value;
      if (json_string[i] == '"') {
        value = parse_string(i);
      } else {
        // Unquoted value (collect until comma or brace)
        size_t start = i;
        while (i < len && json_string[i] != ',' && json_string[i] != '}') {
          i++;
        }
        value = trim(json_string.substr(start, i - start));
      }

      if (!key.empty()) {
        result[key] = value;
      }
    }
  }
} // namespace HTTP
