#ifndef LWM_MANAGER_HH
#define LWM_MANAGER_HH

#include "job.hh"
#include "launcher.hh"
#include "../parallel/threadpool.hh"
#include "../server/server.hh"

#include "spdlog/sinks/stdout_color_sinks.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace LWM {
  using ParallelJobSteps = std::vector<JobStep>;
  using WfDefinition = ParallelJobSteps;
  //using WfDefinition = std::variant<JobStep, ParallelJobSteps>;

  class Manager {
  public:
    Manager() = delete;
    Manager(const std::string& host, std::uint16_t port, LauncherType launch_type);
    Manager(WfDefinition wf_defn, LauncherType launcher_type);

    void queue_workflow(const WfDefinition& wf_defn);
    std::string run_workflow();

  private:
    std::string m_host;
    std::uint16_t m_port;

    HTTP::Server m_server;
    std::unique_ptr<Launcher> m_launcher;
    ThreadPool m_job_pool;

    std::vector<WfDefinition> m_workflows;

    void recurse_workflow(const WfDefinition& wf, MaybeJobFutures_t wait_for);
    std::shared_ptr<std::vector<std::shared_future<JobReturn>>> m_all_futures;

    std::shared_ptr<spdlog::logger> m_logger;
  };

} // namespace LWM
#endif
