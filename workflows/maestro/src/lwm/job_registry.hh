#ifndef LWM_JOB_REGISTRY_HH
#define LWM_JOB_REGISTRY_HH

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace LWM {
  /**
   * Thread-safe registry for tracking active jobs (especially SLURM jobs)
   * to allow for cancellation upon interruption.
   */
  class JobRegistry {
  public:
    static void register_job(const std::string& task_name,
                             const std::string& job_id,
                             const std::string& logfile,
                             std::function<bool(const std::string&)> status_func,
                             std::function<void(const std::string&)> cancel_func);

    static void unregister_job(const std::string& task_name);

    static void cancel_all();

    static void cancel_one(const std::string& jobid);

    static bool is_empty();

    static std::string get_log_file_for(const std::string& job_id);

    static std::chrono::time_point<std::chrono::steady_clock>
    get_last_update_time(const std::string& job_id);

    static void
    set_last_update_time(const std::string& job_id,
                         std::chrono::time_point<std::chrono::steady_clock> update_time);

  private:
    static std::mutex m_registry_mut;
    // Set of (task_name, job_id) pairs
    static std::set<std::pair<std::string, std::string>> m_active_jobs;
    static std::map<std::string,
                    std::chrono::time_point<std::chrono::steady_clock>> m_last_updated;

    // Registry of log files for jobids - jobid: logfile
    static std::map<std::string, std::string> m_job_logfiles;

    // Callbacks registered to get status - Currently simply returns true if running
    // status_func(jobid) -> true if running, false otherwise
    static std::map<std::string, std::function<bool(const std::string&)>> m_status_funcs;

    // Callbacks registered to cancel a job
    // cancel_func(jobid) -> cancels job and returns nothing.
    static std::map<std::string, std::function<void(const std::string&)>> m_cancel_funcs;
  };
} // namespace LWM

#endif
