#include "lwm/manager.hh"

#include "lwm/job.hh"
#include "lwm/job_registry.hh"
#include "lwm/launcher.hh"
#include "util/uuid.hh"

#include "spdlog/cfg/env.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
  inline std::string make_workflow_id() {
    return "lute_wf_" + LWM::uuid_v7();
  }
}

namespace LWM {
  Manager::Manager(const ManagerParameters& params)
    : m_params(params)
    , m_server(HTTP::Server(params.host, params.port))
    , m_job_pool(params.num_manager_threads)
    , m_logger([] {
      if (auto tmp = spdlog::get("LWM:Manager")) {
        return tmp;
      } else {
        return spdlog::stdout_color_mt("LWM:Manager");
      }
    }())
  {
    spdlog::cfg::load_env_levels("LUTE_MAESTRO_LOG_LEVEL");

    m_logger->info("Running workflows with {}",
                   params.launch_type == LauncherType::SlurmLauncherType
                     ? "SlurmLauncher"
                     : "PythonLauncher");

    switch (params.launch_type) {
    case LauncherType::PythonLauncherType: {
      m_launcher = std::move(std::make_unique<PythonLauncher>(params.unbuffered_logs));
      break;
    }
    case LauncherType::SlurmLauncherType: {
      m_launcher = std::move(std::make_unique<SlurmLauncher>(params.unbuffered_logs));
      break;
    }
    }
    m_all_futures =
        std::make_shared<std::vector<std::shared_future<JobReturn>>>();

    if (m_launcher->use_server()) {
      auto request_handlers = m_launcher->get_request_handlers();
      for (auto [route_method_pair, handler] : request_handlers) {
        m_server.add_request_handler(route_method_pair.first,
                                     route_method_pair.second, handler);
      }
    }
  }

  Manager::Manager(WfDefinition wf_defn, LauncherType launch_type)
    : Manager(ManagerParameters(5, 2, false, "0.0.0.0", 8080, launch_type))
  {
    queue_workflow(wf_defn, m_params.ctx);
  }

  Manager::Manager(const std::string& host,
                   std::uint16_t port,
                   LauncherType launch_type)
    : Manager(ManagerParameters(5, 2, false, host, port, launch_type))
  {}

  std::string Manager::queue_workflow(const WfDefinition& wf_defn, LaunchContext ctx) {
    if (ctx.workflow_id.empty()) {
      ctx.workflow_id = make_workflow_id();
    }

    m_workflows.push_back({wf_defn, ctx});

    return ctx.workflow_id;
  }

  void Manager::recurse_workflow(const WfDefinition& wf,
                                 const LaunchContext& ctx,
                                 std::string job_key_prefix,
                                 MaybeJobFutures_t wait_for) {
    // Wrapper launch in lambda (instead of static func)
    auto launch_func = [&](const JobStep& job,
                           const LaunchContext& ctx_,
                           const std::string& job_key,
                           MaybeJobFutures_t wait_for = std::nullopt) -> JobReturn {
      return m_launcher->launch_task(job, ctx_, job_key, wait_for);
    };

    int step_count { 0 };
    for (const auto& step : wf) {
      std::string new_prefix;
      if (job_key_prefix.empty()) {
        new_prefix = std::to_string(step_count);
      } else {
        new_prefix = job_key_prefix + "." + std::to_string(step_count);
      }

      std::string job_key { ctx.workflow_id + "/" + new_prefix };
      auto next_wait_for = std::shared_future<JobReturn>(m_job_pool.enqueue(launch_func,
                                                                            step,
                                                                            ctx,
                                                                            job_key,
                                                                            wait_for));
      m_all_futures->push_back(next_wait_for);
      if (!step.next.empty()) {
        recurse_workflow(step.next,
                         ctx,
                         new_prefix,
                         std::optional<decltype(next_wait_for)>(next_wait_for));
      }
      ++step_count;
    }
  }

  std::string Manager::run_workflow() {
    m_server.start();
    m_logger->info("Beginning workflow.");
    // m_workflows = Vector of Parallel JobSteps
    for (const auto& [wf, ctx] : m_workflows) {
      recurse_workflow(wf, ctx);
      // wf = vector of JobSteps
    }
    int n_complete{0};
    int n_success{0};
    bool cancellation_triggered{false};
    while (true) {
      if (m_all_futures->size() == 0) {
        break;
      }
      if (s_interrupted && !cancellation_triggered) {
        m_logger->warn("Interruption detected! Cancelling active jobs...");
        JobRegistry::cancel_all();
        cancellation_triggered = true;
      }
      if (s_interrupted) {
        m_logger->warn("Interruption detected! Waiting for active tasks to shut down...");
      }
      auto& futures = *m_all_futures;
      for (auto return_it = futures.begin(); return_it != futures.end();) {
        if (return_it->wait_for(std::chrono::milliseconds(200)) == std::future_status::ready) {
          ++n_complete;
          auto& [task_name, status, logfile, splits] = return_it->get();
          std::string msg = "Providing logs for " + task_name + " [Exited as: " + status
            + "]\n" +
            "-----------------------------------------------------------------------\n";
          if (status == "COMPLETED") {
            n_success++;
            m_logger->info(msg);
            using namespace std::literals;
            const auto pending_time = splits.running_point - splits.launch_point;
            const auto running_time = splits.end_point - splits.running_point;
            if (!m_params.unbuffered_logs) {
              std::cout << logfile << std::endl;
            }
            std::cout << "Time " << task_name << " spent: " << std::endl
                      << "- Pending: " << pending_time / 1s << " s" << std::endl
                      << "- Running: " << running_time / 1s << " s"
                      << std::endl;
          } else {
            m_logger->error(msg);
            if (!m_params.unbuffered_logs) {
              std::cout << logfile << std::endl;
            }
          }
          return_it = futures.erase(return_it);
        } else {
          ++return_it;
          m_logger->trace("Waiting on Managed Task completion.");
        }
      }
    }
    m_server.stop();
    if (n_complete == n_success) {
      m_logger->info("Exiting after workflow completed successfully.");
      return "COMPLETED";
    } else {
      m_logger->error(
        "Exiting after workflow failed! "
        "[" + std::to_string(n_success) + "/" + std::to_string(n_complete) + " succeeded]"
      );
      return "FAILED";
    }
  }
} // namespace LWM
