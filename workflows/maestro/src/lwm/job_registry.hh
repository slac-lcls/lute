#ifndef LWM_JOB_REGISTRY_HH
#define LWM_JOB_REGISTRY_HH

#include "lwm/job.hh"

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace LWM {
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  using StatusCallback = std::function<bool(const JobIdentifier&)>;
  using CancelCallback = std::function<void(const JobIdentifier&)>;

  class JobNotRegistered : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
  };

  /**
   * Thread-safe registry for tracking active jobs (especially SLURM jobs)
   * to allow for cancellation upon interruption.
   */
  class JobRegistry {
  public:
    static void register_job(const JobIdentifier& job_id,
                             const std::string& logfile,
                             StatusCallback status_func,
                             CancelCallback cancel_func);

    /**
     * Remove a job from the active job list.
     */
    static void unregister_job(const JobIdentifier& job_id);

    /**
     * Cancel all jobs.
     */
    static void cancel_all();

    /**
     * Cancel one job.
     */
    static void cancel_one(const JobIdentifier& job_id);

    static bool is_empty();

    /**
     * @param[in] job_id The identifier for the job of interest.
     * @returns The path of the log file for the requested job.
     */
    static std::string get_log_file_for(const JobIdentifier& job_id);

    /**
     * @param[in] job_id The identifier for the job of interest.
     * @returns The last time a job provided an update.
     */
    static TimePoint get_last_update_time(const JobIdentifier& job_id);

    /**
     * For a job_key return the complete JobIdentifier (or throw).
     *
     * @note The job_key portion of the JobIdentifier is unique across all jobs.
     *
     * @returns The job_id for a given key.
     * @throws JobNotRegistered if the job_key is not registered as active.
     */
    static const JobIdentifier& job_id_from_key(const std::string& job_key);
    static std::set<JobIdentifier> job_ids_from_name(const std::string& name);

    /**
     * Set the last time a job ID was updated.
     *
     * @param[in] job_id The job.
     * @param[in] update_time The TimePoint of last update.
     */
    static void set_last_update_time(const JobIdentifier& job_id, TimePoint update_time);

    /**
     * Set the update for a job using only the launch_id component of a JobIdentifier.
     *
     * E.g., lookup the job in the registry by a process or SLURM job ID rather than
     * the complete maestro JobIdentifier.
     */
    static void set_last_update_time_by_launch(const std::string& launch_id,
                                               TimePoint update_time);

  private:
    static std::mutex m_registry_mut;

    /**
     * Set of active job IDs.
     */
    static std::set<JobIdentifier> m_active_jobs;
    static std::map<JobIdentifier, TimePoint> m_last_updated;

    /**
     * Registry of log files for job IDs.
     */
    static std::map<JobIdentifier, std::string> m_job_logfiles;

    /**
     * Callbacks registered to get status.
     *
     * status_func(job_id) -> bool.
     *
     * @note Currently just returns `true` if running.
     */
    static std::map<JobIdentifier, StatusCallback> m_status_funcs;

    /**
     * Callbacks registered to cancel a job.
     *
     * cancel_func(job_id) -> void;
     */
    static std::map<JobIdentifier, CancelCallback> m_cancel_funcs;
  };
} // namespace LWM

#endif
