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
  std::set<std::pair<std::string, std::string>> JobRegistry::m_active_jobs;
  std::map<std::string, std::string> JobRegistry::m_job_logfiles;
  std::map<std::string,
           std::chrono::time_point<std::chrono::steady_clock>> JobRegistry::m_last_updated;

  std::map<std::string, std::function<bool(const std::string&)>> JobRegistry::m_status_funcs;
  std::map<std::string, std::function<void(const std::string&)>> JobRegistry::m_cancel_funcs;

  void JobRegistry::register_job(const std::string& task_name,
                                 const std::string& job_id,
                                 const std::string& logfile,
                                 std::function<bool(const std::string&)> status_func,
                                 std::function<void(const std::string&)> cancel_func) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    m_active_jobs.insert({task_name, job_id});
    m_job_logfiles[job_id] = logfile;
    m_last_updated[job_id] = std::chrono::steady_clock::now();

    m_status_funcs[job_id] = status_func;
    m_cancel_funcs[job_id] = cancel_func;
  }

  void JobRegistry::unregister_job(const std::string& task_name) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    for (auto it = m_active_jobs.begin(); it != m_active_jobs.end();) {
      if (it->first == task_name) {
        it = m_active_jobs.erase(it);
      } else {
        ++it;
      }
    }
  }

  void JobRegistry::cancel_all() {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    if (m_active_jobs.empty()) {
      return;
    }

    for (const auto& [name, jobid] : m_active_jobs) {
      auto kill_func = m_cancel_funcs[jobid];
      kill_func(jobid);
    }

    std::string cancel_cmd = "scancel";
    for (const auto& pair : m_active_jobs) {
      cancel_cmd += " " + pair.second;
    }
    // We use std::system here as a quick way to fire and forget the scancel
    // Since we are likely in a signal handler or exiting, we don't want to get
    // too fancy
    int ret = std::system(cancel_cmd.c_str());
    (void)ret;
    m_active_jobs.clear();
  }

  void JobRegistry::cancel_one(const std::string& jobid) {
    if (m_cancel_funcs.count(jobid)) {
      auto kill_func = m_cancel_funcs[jobid];

      kill_func(jobid);
    } else {
      throw std::runtime_error("No registerd cancel operation for job with ID: " + jobid);
    }
  }

  bool JobRegistry::is_empty() {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    return m_active_jobs.empty();
  }

  std::chrono::time_point<std::chrono::steady_clock>
  JobRegistry::get_last_update_time(const std::string& job_id) {
    {
      std::lock_guard<std::mutex> lock(m_registry_mut);
      if (m_last_updated.count(job_id)) {
        return m_last_updated[job_id];
      }
    }

    throw std::runtime_error("Could not find last update time for job: " + job_id);
  }

  void JobRegistry::set_last_update_time(const std::string& job_id,
                                         std::chrono::time_point<std::chrono::steady_clock> update_time) {
    std::lock_guard<std::mutex> lock(m_registry_mut);
    m_last_updated[job_id] = update_time;
  }

  std::string JobRegistry::get_log_file_for(const std::string& job_id) {
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

    throw std::runtime_error("Unable to find log file for Job: " + job_id);
  }
} // namespace LWM
