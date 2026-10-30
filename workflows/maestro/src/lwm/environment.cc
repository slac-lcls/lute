#include "lwm/environment.hh"

#include "lwm/job.hh"

#include <unistd.h>

#include <set>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace LWM {
  std::vector<std::string> setup_environment(const std::vector<std::string>& ctx_env_vars,
                                             const LaunchEnvPolicy env_policy) {
    if (env_policy == LaunchEnvPolicy::CTX_REPLACE) {
      return ctx_env_vars;
    }

    // env_vars containts KEY=VALUE strings. Separate out the names
    std::set<std::string> var_names;
    if (env_policy == LaunchEnvPolicy::CTX_MERGE) {
      for (const auto& key_val : ctx_env_vars) {
        var_names.insert(key_val.substr(0, key_val.find('=')));
      }
    }

    // Merge the provided variables with the current environment
    // Pre-existing are overridden.
    std::vector<std::string> env_out;
    for (char** env_var = environ; env_var && *env_var; ++env_var) {
      std::string_view key_val { *env_var };
      if (env_policy == LaunchEnvPolicy::CTX_IGNORE) {
        env_out.emplace_back(key_val);
      } else {
        auto key { key_val.substr(0, key_val.find('=')) };
        if (!var_names.count(std::string(key))) {
          env_out.emplace_back(key_val);
        }
      }
    }
    if (env_policy == LaunchEnvPolicy::CTX_MERGE) {
      env_out.insert(env_out.end(), ctx_env_vars.begin(), ctx_env_vars.end());
    }

    return env_out;
  }
} // namespace LWM
