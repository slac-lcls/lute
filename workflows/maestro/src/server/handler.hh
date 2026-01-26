#ifndef HTTP_HANDLER_HH
#define HTTP_HANDLER_HH

#include "http.hh"

#include <map>
#include <string>

namespace HTTP {
  class Handler {
  public:
    Handler() = default;
    // Sub-classes may potentially be used via pointers to base so virtual destructor
    virtual ~Handler() = default;

    virtual Response operator()(const HTTP::Request& request) = 0;
  };

  class JsonHandler : public Handler {
  public:
    JsonHandler() = default;
    virtual ~JsonHandler() = default;

    virtual Response operator()(const HTTP::Request& request) = 0;

  protected:
    void parse_json(const std::string& json_string,
                    std::map<std::string, std::string>& result);

  private:
    std::string trim(const std::string& str) const;
  };
} // namespace HTTP

#endif
