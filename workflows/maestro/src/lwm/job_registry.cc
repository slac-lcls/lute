#include "lwm/job_registry.hh"

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace LWM {
  std::mutex JobRegistry::m_registry_mut;

  std::set<JobIdentifier> JobRegistry::m_active_jobs;

  std::map<JobIdentifier, std::string> JobRegistry::m_job_logfiles;

  std::map<JobIdentifier, TimePoint> JobRegistry::m_last_updated;

  std::map<JobIdentifier, StatusCallback> JobRegistry::m_status_funcs;
  std::map<JobIdentifier, CancelCallback> JobRegistry::m_cancel_funcs;

  void JobRegistry::register_job(const JobIdentifier& job_id,
                                 const std::string& logfile,
                                 StatusCallback status_func,
                                 CancelCallback cancel_func) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    m_active_jobs.insert(job_id);
    m_job_logfiles[job_id] = logfile;
    m_last_updated[job_id] = std::chrono::steady_clock::now();

    m_status_funcs[job_id] = status_func;
    m_cancel_funcs[job_id] = cancel_func;
  }

  void JobRegistry::unregister_job(const JobIdentifier& job_id) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    auto it { m_active_jobs.find(job_id) };
    if (it != m_active_jobs.end()) {
      m_active_jobs.erase(it);
    }
  }

  void JobRegistry::cancel_all() {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    for (const auto& job_id : m_active_jobs) {
      auto kill_func_it { m_cancel_funcs.find(job_id) };
      if (kill_func_it != m_cancel_funcs.end()) {
        kill_func_it->second(job_id);
      }
    }

    m_active_jobs.clear();
  }

  void JobRegistry::cancel_one(const JobIdentifier& job_id) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    if (m_cancel_funcs.count(job_id)) {
      auto kill_func = m_cancel_funcs[job_id];

      kill_func(job_id);
    } else {
      throw std::runtime_error("No registered cancel operation for job: " + job_id.job_key);
    }
  }

  bool JobRegistry::is_empty() {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    return m_active_jobs.empty();
  }

  TimePoint JobRegistry::get_last_update_time(const JobIdentifier& job_id) {
    {
      std::lock_guard<std::mutex> lock(m_registry_mut);
      if (m_last_updated.count(job_id)) {
        return m_last_updated[job_id];
      }
    }

    throw std::runtime_error("Could not find last update time for job: " + job_id.job_key);
  }

  const JobIdentifier& JobRegistry::job_id_from_key(const std::string& job_key) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    for (const auto& job_id : m_active_jobs) {
      if (job_id.job_key == job_key) {
        return job_id;
      }
    }

    throw JobNotRegistered("Job key " + job_key + " has not been registered!");
  }

  std::set<JobIdentifier> JobRegistry::job_ids_from_name(const std::string& name) {
    std::set<JobIdentifier> job_ids;

    for (const auto& job_id : m_active_jobs) {
      if (job_id.name.has_value() && *(job_id.name) == name) {
        job_ids.insert(job_id);
      }
    }

    return job_ids;
  }

  void JobRegistry::set_last_update_time(const JobIdentifier& job_id, TimePoint update_time) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    m_last_updated[job_id] = update_time;
  }

  void JobRegistry::set_last_update_time_by_launch(const std::string& launch_id,
                                                   TimePoint update_time) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    for (const auto& job_id : m_active_jobs) {
      if (job_id.launch_id == launch_id) {
        m_last_updated[job_id] = update_time;
      }
    }
  }

  std::string JobRegistry::get_log_file_for(const JobIdentifier& job_id) {
    std::string logfile { "" };
    {
      std::lock_guard<std::mutex> lock(m_registry_mut);
      if (m_job_logfiles.count(job_id)) {
        logfile = m_job_logfiles[job_id];
      }
    }

    if (!logfile.empty()) {
      return logfile;
    }

    throw std::runtime_error("Unable to find log file for Job: " + job_id.job_key);
  }
} // namespace LWM
