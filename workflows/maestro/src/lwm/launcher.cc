#include "launcher.hh"

#include "job.hh"

#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream> // Consider removing this later!
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

namespace LWM {

  void JsonStatusHandler::update_running_splits(JobStepSplits* splits,
                                                std::string& managed_task_name) {
    std::lock_guard<std::mutex> lock(m_status_mut);
    m_splits_map[managed_task_name] = splits;
  }

  void JsonStatusHandler::remove_running_splits(std::string& managed_task_name) {
    std::lock_guard<std::mutex> lock(m_status_mut);
    if (m_splits_map.find(managed_task_name) != m_splits_map.end()) {
      m_splits_map.erase(managed_task_name);
    }
  }

  HTTP::Response JsonStatusHandler::operator()(const HTTP::Request& request) {
    std::map<std::string,std::string> status;
    parse_json(request.content(), status); // Implemented by HTTP::JsonHandler
    {
      std::lock_guard<std::mutex> lock(m_status_mut);
      m_status_map[status["managed_task"]] = status["status"];
      if (status["status"] == "STARTED" &&
          m_splits_map.find(status["managed_task"]) != m_splits_map.end()) {
        m_splits_map[status["managed_task"]]->running_point = std::chrono::steady_clock::now();
      }
    }

    HTTP::Response response(HTTP::CODE::OK);
    response.set_content("Status received.");
    return response;
  }

  HTTP::Response JsonLogHandler::operator()(const HTTP::Request &request) {
    std::map<std::string, std::string> log;
    if (m_unbuffered_logs) {
      parse_json(request.content(), log);
      {
    //  std::lock_guard<std::mutex> lock(m_log_mut);
    //  m_log_map[log["managed_task"]] = log["message"];
        if (log.find("managed_task") != log.end()) {
          std::string msg = "[";
          msg += log["managed_task"] + "] ";
          if (log.find("log") != log.end()) {
            m_logger->info(msg + log["log"]);
          } else if (log.find("message") != log.end()) {
            m_logger->info(msg + log["message"]);
          }
        }
      }
    }
    HTTP::Response response(HTTP::CODE::OK);
    response.set_content("Log received.");
    return response;
  }

  bool Launcher::can_task_run(const JobStep& job, MaybeJobFutures_t wait_for, std::string& reason) {
    TriggerRule rule = job.trigger_rule;
    if (wait_for) {
      // Have upstream jobs to wait on, either 1 or a vector
      JobFutures_t job_futures = *wait_for;
      if (std::holds_alternative<std::vector<JobFuture_t>>(job_futures)) {
        m_logger->debug(job.managed_task_name + " is waiting on a multiple jobs.");
        auto& futures = std::get<std::vector<JobFuture_t>>(job_futures);
        int n_complete{0};
        int n_success{0};
        while (true) {
          if (!futures.size()) {
            break;
          }
          for (auto return_it = futures.begin(); return_it != futures.end(); ) {
            if (return_it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
              if (rule == TriggerRule::ALWAYS) {
                return true;
              }
              std::string status = return_it->get().status;
              if (status == "COMPLETED") {
                reason = std::string("A previous job succeeded and have rule: ") +
                         rule_to_string(rule);
                if (rule == TriggerRule::ANY_SUCCESS) {
                  return true;
                } else if (rule == TriggerRule::ALL_FAILED) {
                  return false;
                }
                ++n_success;
              } else { // All other statuses indicate failure
                reason = std::string("A previous job failed and have rule: ") +
                         rule_to_string(rule);
                if (rule == TriggerRule::ALL_SUCCESS) {
                  return false;
                } else if (rule == TriggerRule::ANY_FAILED) {
                  return true;
                }
              }
              ++n_complete;
              futures.erase(return_it);
            } else {
              ++return_it;
            }
          }
          std::this_thread::yield();
        }
        if (rule == TriggerRule::ALL_COMPLETED) {
          reason = std::string("All previous jobs completed and have rule: ") +
                   rule_to_string(rule);
          return true;
        } else if (n_complete == n_success) {
          reason = std::string("All previous jobs succeeded and have rule: ") +
                   rule_to_string(rule);
          if (rule == TriggerRule::ALL_SUCCESS) {
            return true;
          }
          return false;
        } else {
          // This should be the case n_success == 0
          reason = std::string("All previous jobs failed and have rule: ") +
                   rule_to_string(rule);
          if (rule == TriggerRule::ALL_FAILED) {
            return true;
          }
          return false;
        }
      } else {
        m_logger->debug(job.managed_task_name + " is waiting on a job.");
        auto& prev_job_future = std::get<JobFuture_t>(job_futures);
        std::string status = prev_job_future.get().status;
        if (status == "COMPLETED") { // Only successful outcome
          reason = std::string("All previous jobs succeeded and have rule: ") +
                   rule_to_string(rule);
          if (rule != TriggerRule::ALL_FAILED && rule != TriggerRule::ANY_FAILED) {
            return true;
          }
          return false;
        } else {
          reason = std::string("All previous jobs failed and have rule: ") +
                   rule_to_string(rule);
          if (rule == TriggerRule::ALL_SUCCESS || rule == TriggerRule::ANY_SUCCESS) {
            // Return false on ANY_SUCCESS because this is the only upstream
            // No possible success will come.
            return false;
          }
          return true;
        }
      }
    } else {
      reason = std::string("No jobs to wait on.");
      // No previous job to wait on
      return true;
    }
    return true; // Broke out of all other cass.
  }

  std::pair<std::string,int> SubprocessLauncher::run_subprocess_log(const std::string& cmd,
                                                                    bool return_output) {
    std::string final_cmd = cmd;
    std::FILE* tmp_file;
    std::string tmp_filename;
    if (return_output) {
      tmp_file = std::tmpfile();
      if (!tmp_file) {
        m_logger->critical("Cannot create a temporary file!");
        throw std::runtime_error("Cannot create a temporary file.");
      }
      tmp_filename = fs::read_symlink(
        fs::path("/proc/self/fd") / std::to_string(fileno(tmp_file))
      );
      // This includes a `(deleted)` in the name
      std::regex deleted_pattern("\\s*\\(deleted\\)");
      tmp_filename = std::regex_replace(tmp_filename, deleted_pattern, "");
      m_logger->trace("Will redirect subprocess logs to: " + tmp_filename);
      final_cmd = cmd + " > " + tmp_filename + " 2>&1";
    }
    int status = std::system(final_cmd.c_str());

    if (WEXITSTATUS(status)) {
      std::string msg{
        "Error running subprocess. Return code: " + std::to_string(WEXITSTATUS(status))
      };
      m_logger->error(msg);
    }

    if (!return_output) {
      return std::make_pair("",status);
    } else {
      std::string result;
      std::ifstream in(tmp_filename);
      if (in.is_open()) {
        result.assign((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
        in.close();
      } else {
        std::string err =
          "Unable to open temporary command output file: ";
        err += tmp_filename;
        throw std::runtime_error(err.c_str());
      }
      return std::make_pair(result, status);
    }
    return std::make_pair("", status);
  }

  JobReturn SubprocessLauncher::launch_task(const JobStep& job, bool is_daq2, MaybeJobFutures_t wait_for) {
    JobStepSplits splits;
    std::string managed_task_name = job.managed_task_name;
    std::string msg{"Preparing to run: "};
    std::string reason;
    msg += managed_task_name;
    logger()->debug(msg);
    std::string log, status;
    if (can_task_run(job, wait_for, reason)) { // This will block on futures as needed
      std::string launch_cmd = prepare_launch_cmd(job, is_daq2);
      msg = "Will launch " + managed_task_name + " with: " + launch_cmd;
      logger()->info(msg);
      if (launch_cmd.empty()) {
        std::string err_msg = "Unable to create a command string for " +
                              job.managed_task_name +
                              " the return string was empty!";
        throw std::runtime_error(err_msg.c_str());
      }
      m_status_handler->update_running_splits(&splits, managed_task_name);
      splits.launch_point = std::chrono::steady_clock::now();
      int ret_status;
      std::tie(log, ret_status) = run_subprocess_log(launch_cmd, true);
      if (ret_status != 0) {
        status = "SUBPROCESS_FAILED";
        return std::move(JobReturn(managed_task_name, status, log, splits));
      }
      std::regex jobid_regex(R"(Submitted batch job ([0-9]{0,100}))");
      std::smatch jobid_match;
      std::string jobid{""};
      if (std::regex_search(log,jobid_match,jobid_regex)) {
        jobid = jobid_match[1].str();
      }

      while (status != "COMPLETED" && status != "FAILED" && status != "CANCELLED" && status != "TIMEDOUT") {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        // ... Need to do the status check ... //
        {
          std::lock_guard<std::mutex>(m_status_handler->m_status_mut);
          for (auto& pair : m_status_handler->m_status_map) {
            if (pair.first == managed_task_name) {
              status = pair.second;
            }
          }
        }
        logger()->debug(managed_task_name + "'s current status is " + status);
      }
      status = "COMPLETED";
      update_log(log, jobid);
    } else {
      // TODO: Need to change this to update the message to acocunt for trigger
      // rule and explain why not running
      log = "---- NOT LAUNCHING " + job.managed_task_name + " ----\n" +
            reason + "\n";
      status = "UPSTREAM_FAILED";
    }
    splits.end_point = std::chrono::steady_clock::now();
    m_status_handler->remove_running_splits(managed_task_name);
    return std::move(JobReturn(managed_task_name, status, log, splits));
  }

  JobReturn SubprocessLauncher::operator()(const JobStep& job, bool is_daq2, MaybeJobFutures_t wait_for) {
    return launch_task(job, is_daq2, wait_for);
  }

  std::string SlurmLauncher::prepare_launch_cmd(const JobStep& job, bool is_daq2) {
    //if (job.parameters.executable_subdir == "launch_")
    std::string executable =
      job.parameters.lute_location + "/" + job.parameters.executable_subdir
      + "/submit_slurm.sh";

    std::string config_file = job.parameters.config_file;
    std::string param_str =
        "--taskname " + job.managed_task_name + " --config " + config_file;

    if (job.parameters.debug) {
      param_str += " --debug";
    }
    if (is_daq2) {
      param_str += " --psana2";
    }

    std::string slurm_params = job.extra_parameters;

    param_str += " " + slurm_params;

    return executable + " " + param_str;
  }

  void SlurmLauncher::update_log(std::string& log, std::string& jobid) {
    if (jobid.empty()) {
      logger()->error("Trying to get information for an empty SLURM jobid!");
    }
    std::string get_logfile_cmd{"sacct -j " + jobid + " -o StdOut%200"};
    auto [slurm_info, ret_code] = run_subprocess_log(get_logfile_cmd, true);

    std::regex logfile_regex(R"((/[^\s]+\.out))");
    std::smatch logfile_match;
    std::string logfile_path{""};
    if (std::regex_search(slurm_info, logfile_match, logfile_regex)) {
      logfile_path = logfile_match[1].str();
    } else {
      return;
    }

    size_t pos = logfile_path.find("%J");
    if (pos != std::string::npos) {
      logfile_path.replace(pos, 2, jobid);
    } else {
      return;
    }

    std::string get_slurm_log_cmd{"cat "+logfile_path};
    std::tie(log, ret_code) = run_subprocess_log(get_slurm_log_cmd, true);
  }

  std::string PythonLauncher::prepare_launch_cmd(const JobStep& job, bool is_daq2) {
    std::string executable = "python ";

    bool debug = job.parameters.debug;
    executable += debug ? "-B " : "-OB ";

    std::string script_location;
    if (job.parameters.executable_subdir == "launch_scripts") {
      script_location = job.parameters.lute_location + "/run_task.py ";
    } else {
      script_location = job.parameters.lute_location + "/" + job.parameters.executable_subdir + "/run_task ";
    }

    executable += script_location;

    std::string config_file = job.parameters.config_file;
    std::string param_str =
        "--taskname " + job.managed_task_name + " --config " + config_file;

    return executable + " " + param_str;
  }

} //namespace LWM
