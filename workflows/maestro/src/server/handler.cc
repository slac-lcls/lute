#include "handler.hh"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <string>

namespace HTTP {
  std::string JsonHandler::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    size_t last = str.find_last_not_of(" \t\r\n");
    return (first == std::string::npos || last == std::string::npos)
      ? ""
      : str.substr(first, last - first + 1);
  }

  void JsonHandler::parse_json(const std::string& json_string,
                               std::map<std::string, std::string>& result) {

    rapidjson::Document json;
    json.Parse(json_string.c_str());
    if (json.HasParseError()) {
      result["JSON_PARSE_ERROR"] = "true";
      return;
    }
    for (auto member = json.MemberBegin(); member != json.MemberEnd(); ++member) {
      std::string key {member->name.GetString()};
      std::string val;
      // We don't do multiple layers at the moment. If its not an actual string
      // we'll just put it back in as a string
      if (member->value.IsString()) {
        val = member->value.GetString();
      } else {
        // Serialize it back into a string
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        member->value.Accept(writer);
        val = buffer.GetString();
      }
      result[key] = val;
    }
  }
} // namespace HTTP
