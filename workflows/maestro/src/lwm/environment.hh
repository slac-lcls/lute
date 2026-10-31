#ifndef LWM_ENVIRONMENT_HH
#define LWM_ENVIRONMENT_HH

#include "lwm/job.hh"

#include <string>
#include <vector>

namespace LWM {
  std::vector<std::string> setup_environment(const std::vector<std::string>& ctx_env_vars,
                                             const LaunchEnvPolicy env_policy);
} // namespace LWM

#endif // LWM_ENVIRONMENT_HH
