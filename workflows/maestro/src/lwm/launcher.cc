#include "lwm/launcher.hh"

#include "lwm/job.hh"
#include "lwm/job_registry.hh"

#if defined(__linux__) || defined(__APPLE__) || defined(__OSX__) || defined(__APPLE_CC__)
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#else
#include <windows.h>
#endif
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

namespace {
  /**
   * RAII helper for temporary files.
   * Uses mkstemp for unique filename generation and ensures the file is
   * deleted upon destruction.
   */
  struct ScopedTempFile {
    fs::path path;
    int fd { -1 };

    ScopedTempFile() {
      char temp_template[] = "/tmp/lute_maestro_XXXXXX";
      fd = mkstemp(temp_template);
      if (fd == -1) {
        throw std::runtime_error("Failed to create temporary file using mkstemp");
      }
      path = temp_template;
      // We close the descriptor immediately as we only need the unique filename
      // for shell redirection. The file exists now with 0600 permissions.
      close(fd);
      fd = -1;
    }

    ~ScopedTempFile() {
      if (fd != -1) {
        close(fd);
      }

      if (!path.empty()) {
        std::error_code ec;
        fs::remove(path, ec);
      }
    }
  };
} // anonymous namespace

namespace LWM {
  std::atomic<bool> s_interrupted { false };

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
    std::map<std::string,std::string> status_plus_data;
    parse_json(request.content(), status_plus_data); // Implemented by HTTP::JsonHandler

    // Check for JobID and update the most recent time stamp
    if (status_plus_data.count("orig_job_id")) {
      std::string jobid { status_plus_data["orig_job_id"] };
      auto ts { std::chrono::steady_clock::now() };
      JobRegistry::set_last_update_time(jobid, ts);
    }

    if (status_plus_data.count("managed_task")) {
      std::lock_guard<std::mutex> lock(m_status_mut);
      std::string task_name = status_plus_data["managed_task"];
      if (status_plus_data.count("status")) {
        m_status_map[task_name] = status_plus_data["status"];
        if (status_plus_data["status"] == "STARTED" && m_splits_map.count(task_name)) {
          m_splits_map[status_plus_data["managed_task"]]->running_point = std::chrono::steady_clock::now();
        }
      }

      // Store additional fields as metadata
      for (auto const& [key, val] : status_plus_data) {
        if (key != "managed_task" && key != "status") {
          m_metadata_map[task_name][key] = val;
        }
      }
    }

    HTTP::Response response(HTTP::CODE::OK);
    response.set_content("Status received.");
    return response;
  }

  HTTP::Response JsonLogHandler::operator()(const HTTP::Request& request) {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string GREEN = "\033[32m";
    std::map<std::string, std::string> log;
    if (m_unbuffered_logs) {
      parse_json(request.content(), log);
      {
        // Check for JobID and update the most recent time stamp
        if (log.count("orig_job_id")) {
          std::string jobid { log["orig_job_id"] };
          auto ts { std::chrono::steady_clock::now() };
          JobRegistry::set_last_update_time(jobid, ts);
        }
        //  std::lock_guard<std::mutex> lock(m_log_mut);
        //  m_log_map[log["managed_task"]] = log["message"];
        if (log.find("managed_task") != log.end()) {
          std::string msg {
            "\n\t" + BOLD + GREEN + "[Logs from **" + log["managed_task"] + "**]" + RESET + "\n"
          };
          if (log.find("log") != log.end()) {
            msg += log["log"];
          } else if (log.find("message") != log.end()) {
            msg += log["message"];
          }
          m_logger->info(msg);
        }
      }
    }
    HTTP::Response response(HTTP::CODE::OK);
    response.set_content("Log received.");
    return response;
  }

  HTTP::Response JsonTasksHandler::operator()(const HTTP::Request& request) {
    std::string json_res = "{ \"managed_tasks\": [";
    {
      std::lock_guard<std::mutex> lock(m_status_handler->m_status_mut);
      bool first = true;

      // I kinda regret how I set this up now... but will recombine
      // stuff into a single map to do the JSON string conversion
      // TODO: Probably need to restructure all this key/val handling to be smarter...
      for (const auto& [managed_task_name, status] : m_status_handler->m_status_map) {
        if (!first) {
          json_res += ", ";
        }
        std::map<std::string, std::string> json_map;
        json_map["name"] = managed_task_name;
        json_map["status"] = status;
        if (m_status_handler->m_metadata_map.count(managed_task_name)) {
          const auto& task_meta = m_status_handler->m_metadata_map[managed_task_name];
          json_map.insert(task_meta.begin(), task_meta.end());
        }
        try {
          json_res += to_json_str(json_map);
          first = false;
        } catch (const HTTP::InvalidJsonMaps& e) {
          m_logger->error("Can't JSONify data for: " + managed_task_name + ". " + e.what());
        }
      }
    }
    json_res += "] }";
    m_logger->debug("Created response: {}", json_res);
    HTTP::Response response(HTTP::CODE::OK);
    response.set_header("Content-Type", "application/json");
    response.set_content(json_res);
    return response;
  }

  HTTP::Response JsonRpcHandler::operator()(const HTTP::Request& request) {
    if (request.method() == HTTP::METHOD::POST) {
      std::map<std::string, std::string> rpc_data;
      parse_json(request.content(), rpc_data);

      // Check for JobID and update the most recent time stamp
      if (rpc_data.count("orig_job_id")) {
        std::string jobid { rpc_data["orig_job_id"] };
        auto ts { std::chrono::steady_clock::now() };
        JobRegistry::set_last_update_time(jobid, ts);
      }

      if (rpc_data.find("target") != rpc_data.end()
          && rpc_data.find("message") != rpc_data.end()) {
        std::lock_guard<std::mutex> lock(m_status_handler->m_status_mut);
        m_status_handler->m_rpc_queues[rpc_data["target"]].push(rpc_data["message"]);

        HTTP::Response response(HTTP::CODE::OK);
        response.set_content("Message queued.");
        return response;
      }
      return HTTP::Response(HTTP::CODE::BadRequest);
    } else if (request.method() == HTTP::METHOD::GET) {
      // Basic query param parsing for ?task=...
      std::string task_name;
      size_t pos = request.url().find("task=");
      if (pos != std::string::npos) {
        task_name = request.url().substr(pos + 5);
        size_t amp_pos = task_name.find('&');
        if (amp_pos != std::string::npos) {
          task_name = task_name.substr(0, amp_pos);
        }
      } else {
        // Try body if not in query param
        std::map<std::string, std::string> rpc_data;
        parse_json(request.content(), rpc_data);
        if (rpc_data.find("task") != rpc_data.end()) {
          task_name = rpc_data["task"];
        }

        // Check for JobID and update the most recent time stamp
        if (rpc_data.count("orig_job_id")) {
          std::string jobid { rpc_data["orig_job_id"] };
          auto ts { std::chrono::steady_clock::now() };
          JobRegistry::set_last_update_time(jobid, ts);
        }
      }

      if (!task_name.empty()) {
        std::lock_guard<std::mutex> lock(m_status_handler->m_status_mut);
        if (m_status_handler->m_rpc_queues.find(task_name) != m_status_handler->m_rpc_queues.end()
            && !m_status_handler->m_rpc_queues[task_name].empty()) {
          std::string msg = m_status_handler->m_rpc_queues[task_name].front();
          m_status_handler->m_rpc_queues[task_name].pop();

          HTTP::Response response(HTTP::CODE::OK);
          response.set_header("Content-Type", "application/json");
          response.set_content("{ \"message\": \"" + msg + "\" }");
          return response;
        }
      }

      HTTP::Response response(HTTP::CODE::OK);
      response.set_header("Content-Type", "application/json");
      response.set_content("{ \"message\": null }");
      return response;
    }
    return HTTP::Response(HTTP::CODE::MethodNotAllowed);
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
    std::string final_cmd { cmd };
    std::unique_ptr<ScopedTempFile> tmp_file;

    if (return_output) {
      try {
        tmp_file.reset(new ScopedTempFile());
        m_logger->debug("Will redirect subprocess logs to: {}", tmp_file->path.string());
        final_cmd = cmd + " > " + tmp_file->path.string() + " 2>&1";
      } catch (const std::exception& e) {
        m_logger->critical("Cannot create a temporary file: {}", e.what());
        throw;
      }
    }

    int err_pipes[2];
    auto ret { pipe(err_pipes) };
    if (ret == -1) {
      std::string err_result { "Pipe failed - cannot launch subprocess!" };
      m_logger->critical(err_result);

      return std::make_pair(err_result, ret);
    }

    ret = fcntl(err_pipes[1], F_SETFD, FD_CLOEXEC);
    if (ret == -1) {
      std::string err_result { "fcntl failed - cannot launch subprocess!" };
      m_logger->critical(err_result);

      return std::make_pair(err_result, ret);
    }

    pid_t pid = fork();

    if (pid < 0) {
      std::string err_result { "Fork failed - cannot launch subprocess!" };
      m_logger->critical(err_result);

      close(err_pipes[0]);
      close(err_pipes[1]);

      return std::make_pair(err_result, pid);
    } else if (pid == 0) {
      close(err_pipes[0]);

      execl("/bin/sh", "sh", "-c", final_cmd.c_str(), nullptr);

      int err { errno };
      (void)write(err_pipes[1], &err, sizeof(err));

      close(err_pipes[1]);
      _exit(127);
    } else {
      close(err_pipes[1]);

      int status { 0 };

      ssize_t n = read(err_pipes[0], &status, sizeof(status));
      close(err_pipes[0]);

      if (n < 0) {
        std::string err_result { "read failed - cannot track subprocess state!" };
        m_logger->critical(err_result);

        return std::make_pair(err_result, status);
      } else if (n == sizeof(int)) {
        // exec failed, and an int was written with errno
        std::string err_result { "Exec failed with: " + std::to_string(status) };
        m_logger->error(err_result);

        return std::make_pair(err_result, status);
      }
      // else, n == 0, so pipe closed by FD_CLOEXEC, and can now record proc output

      // If we're returning all the output from the launch, wait here.
      // `status` will record the final process exit status.
      // If not returning all the output, then the `status` records that the fork/exec
      // was successful. The newly launched subprocess will use other means to report
      // back as it progresses.
      if (return_output) {
        ret = waitpid(pid, &status, 0);

        if (ret == -1) {
          std::string err_result { "waitpid failed - cannot track subprocess state!" };
          m_logger->critical(err_result);

          return std::make_pair(err_result, ret);
        }

        std::string result;
        std::ifstream in(tmp_file->path);

        if (in.is_open()) {
          result.assign((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
        } else {
          std::string err_result {
            "Unable to open temporary command output file: "
          };

          err_result += tmp_file->path.string();

          return std::make_pair(err_result, -1);
        }

        return std::make_pair(result, status);
      } else {
        return std::make_pair("", status);
      }
    }
  }

  std::optional<std::string>
  SubprocessLauncher::add_job_to_registry(std::string& managed_task_name,
                                          std::string& log) {
    static std::map<std::string, unsigned> task_counts;

    if (task_counts.count(managed_task_name)) {
      task_counts[managed_task_name]++;
    } else {
      task_counts[managed_task_name] = 1;
    }

    return std::make_optional(std::to_string(task_counts[managed_task_name]));
  }

  JobReturn SubprocessLauncher::launch_task(const JobStep& job,
                                            bool is_daq2,
                                            MaybeJobFutures_t wait_for) {
    JobStepSplits splits;
    std::string managed_task_name = job.managed_task_name;
    logger()->debug("Preparing to run: {}", managed_task_name);

    std::string reason;
    std::string log, status;
    if (can_task_run(job, wait_for, reason)) { // This will block on futures as needed
      std::string launch_cmd = prepare_launch_cmd(job, is_daq2);
      logger()->info("Will launch {} with: {}", managed_task_name, launch_cmd);

      if (launch_cmd.empty()) {
        std::string err_msg = "Unable to create a command string for " +
                              job.managed_task_name +
                              " the return string was empty!";
        throw std::runtime_error(err_msg.c_str());
      }

      m_status_handler->update_running_splits(&splits, managed_task_name);
      splits.launch_point = std::chrono::steady_clock::now();

      // run_subprocess_log is modified to not return the actual return code of the
      // subprocess if not tracking the full output (second arg is false). This allows
      // fireing-and-forgetting the subprocess which is important for some launchers
      // e.g. the PythonLauncher - we dont want to stall this process to track it.
      // The log will be collated later (either via the HTTP server, or at the end).
      //
      // If non-zero, it means that the actual launch failed in some manner:
      // - Problem with fork/exec
      // - Or assorted problems with the various fd utilties/fcntl etc.
      // The log will have an exact reason.
      int ret_status;
      //std::tie(log, ret_status) = run_subprocess_log(launch_cmd, false);
      std::tie(log, ret_status) = run_subprocess_log(launch_cmd, true);
      if (ret_status != 0) {
        m_logger->error("Error submitting job for {}. Could not launch or track subprocess",
                        job.managed_task_name);

        status = "SUBPROCESS_FAILED";

        splits.end_point = std::chrono::steady_clock::now();
        return JobReturn(managed_task_name, status, log, splits);
      }

      auto jobid_opt = this->add_job_to_registry(managed_task_name, log);
      if (!jobid_opt.has_value()) {
        status = "SUBPROCESS_NOT_REGISTERED";

        splits.end_point = std::chrono::steady_clock::now();
        return JobReturn(managed_task_name, status, log, splits);
      }

      std::string jobid { *jobid_opt };
      std::string logfile = JobRegistry::get_log_file_for(jobid);
      logger()->info("Will begin monitoring {}. Individual log file will be at: {}. {}",
                     jobid,
                     logfile,
                     this->m_unbuffered_logs
                     ? "Will stream logs in real time, too."
                     : "Will collect logs here at end, too.");

      while (status != "COMPLETED" && status != "FAILED" && status != "CANCELLED" && status != "TIMEDOUT") {
        if (s_interrupted) {
          status = "CANCELLED";
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // ... Need to do the status check ... //
        {
          std::lock_guard<std::mutex>(m_status_handler->m_status_mut);
          for (auto& pair : m_status_handler->m_status_map) {
            if (pair.first == managed_task_name) {
              status = pair.second;
            }
          }
        }
        logger()->trace(managed_task_name + "'s current status is " + status);

        {
          // Check last update time and ping directly if too long
          using namespace std::literals;
          auto update_pt { JobRegistry::get_last_update_time(jobid) };
          auto now = std::chrono::steady_clock::now();
          const std::chrono::duration<double> elapsed_dur { (now - update_pt) / 1ms };
          const double elapsed { elapsed_dur.count() / 1e3 };

          if (status.empty() || status == "PENDING") {
            // If we're waiting for the job to even start, we'll wait 15 minutes before
            // even pinging database...
            if (elapsed < 900.0) {
              continue;
            }
          }

          if (elapsed > 120.0) {
            // Check status....
            bool is_running = this->is_subprocess_running(jobid);
            if (!is_running) {
              status = "FAILED_UNRESPONSIVE";
              logger()->error("Job exited unexpectedly! No indication of why! Marked failed!");

              break;
            } else {
              // Have additional check to cancel?? Stuck job??
              auto time_for_cancel = std::getenv("LUTE_UNRESP_KILL_TIME");
              if (time_for_cancel) {
                if (elapsed > std::atoi(time_for_cancel)) {
                  JobRegistry::cancel_one(jobid);
                }
              } else if (elapsed > 900.0) { // Use a 15 minute default? Should be reasonable
                JobRegistry::cancel_one(jobid);
              }
            }
          }
        }
      }
      JobRegistry::unregister_job(managed_task_name);
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
    return JobReturn(managed_task_name, status, log, splits);
  }

  JobReturn SubprocessLauncher::operator()(const JobStep& job, bool is_daq2, MaybeJobFutures_t wait_for) {
    return launch_task(job, is_daq2, wait_for);
  }

  bool SlurmLauncher::is_subprocess_running(const std::string& jobid) {
    std::string get_job_state_cmd { "sacct -j " + jobid + " -o start,end,state" };

    m_logger->debug("Checking job state from SLURMDB with: {}", get_job_state_cmd);
    auto [slurm_info, ret_code] = run_subprocess_log(get_job_state_cmd, true);

    std::istringstream input;
    input.str(slurm_info);

    // Expecting a columnar output of this format:
    //              Start                 End      State
    // ------------------- ------------------- ----------
    // YYYY-MM-DDTHH:MM:SS YYYY-MM-DDTHH:MM:SS  <STATUS>+
    // YYYY-MM-DDTHH:MM:SS YYYY-MM-DDTHH:MM:SS   <STATUS>
    // YYYY-MM-DDTHH:MM:SS YYYY-MM-DDTHH:MM:SS   <STATUS>

    // Possible failures are:
    // - CANCELLED
    // - FAILED
    // - DEADLINE
    // - TIMEOUT
    // - BOOT_FAIL
    // - NODE_FAIL
    // - OUT_OF_MEMORY
    // - PREEMPTED
    //
    // Otehrwise if not RUNNING/COMPLETING we have:
    // - COMPLETED

    // The below loop is currently redundant since just looking for a substring above
    // Eventually, though, can use the timestamps we pull out here, or do smarter
    // parsing (e.g. preempted vs timed out vs cancelled and so on).
    // For now, just say its dead or done
    std::string line;
    while (std::getline(input, line)) {
      std::vector<std::string> cols;
      std::size_t pos { 0 };
      std::size_t last_pos { 0 };
      std::string col;
      while ((pos = line.find(' ', last_pos)) != std::string::npos) {
        col = line.substr(last_pos, pos);
        if (col.find("RUNNING") != std::string::npos ||
            col.find("PENDING") != std::string::npos) { // Treat this as running for our purposes
          // Check the common states first, RUNNING, PENDING
          // or below, COMPLETED/CANCELLED/FAILED
          logger()->trace("Job {} marked as running to {}.", jobid, col);
          return true;
        }

        // This is incredibly dumb....
        if (col.find("COMPLETED")     != std::string::npos ||
            col.find("CANCELLED")     != std::string::npos ||
            col.find("FAILED")        != std::string::npos ||
            col.find("BOOT_FAIL")     != std::string::npos ||
            col.find("DEADLINE")      != std::string::npos ||
            col.find("PREEMPTED")     != std::string::npos ||
            col.find("OUT_OF_MEMORY") != std::string::npos ||
            col.find("NODE_FAIL")     != std::string::npos) {
          logger()->debug("Job {} marked as not running due to {}.", jobid, col);

          return false;
        }

        cols.push_back(col);
        last_pos = pos + 1;
      }

      cols.push_back(col);
    }
  }

  void SlurmLauncher::cancel_subprocess(const std::string& jobid) {
    // Will use scancel <jobid> for the cancel operation
    std::string cancel_cmd { "scancel " + jobid };

    // Since it will maybe be called from the JobRegistry, and it could be from
    // inside a signal handler/during rapid teardown, use std::system to just
    // fire it off.

    int ret { std::system(cancel_cmd.c_str()) };
    (void)ret;
  }

  std::optional<std::string>
  SlurmLauncher::add_job_to_registry(std::string& managed_task_name, std::string& log) {
    std::regex jobid_regex(R"(Submitted batch job ([0-9]{0,100}))");
    std::smatch jobid_match;
    std::string jobid { "" };
    if (std::regex_search(log, jobid_match, jobid_regex)) {
      jobid = jobid_match[1].str();

      if (jobid.empty()) {
        logger()->error("Determined SLURM jobid was empty!");

        return std::nullopt;
      }

      // Allow some time for submission and then backoff
      std::size_t wait_ms { 500 };
      bool found_logfile { false };

      std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
      std::string get_logfile_cmd { "sacct -j " + jobid + " -o StdOut%200" };
      m_logger->debug("Retrieving log file path with: {}", get_logfile_cmd);
      auto [slurm_info, ret_code] = run_subprocess_log(get_logfile_cmd, true);

      std::string logfile_path { "" };
      while (!found_logfile && wait_ms < 10000) {
        if (WEXITSTATUS(ret_code)) {
          logger()->error("Unable to run {}. Return code: {}",
                          get_logfile_cmd,
                          std::to_string(WEXITSTATUS(ret_code)));
          return std::nullopt;
        }

        std::regex logfile_regex(R"((/[^\s]+\.out))");
        std::smatch logfile_match;
        if (std::regex_search(slurm_info, logfile_match, logfile_regex)) {
          logfile_path = logfile_match[1].str();
          found_logfile = true;
          break;
        }

        wait_ms <<= 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        std::tie(slurm_info, ret_code) = run_subprocess_log(get_logfile_cmd, true);
      }

      if (!found_logfile) {
        logger()->error("Unable to grab a logfile path from sacct -j. Output was: {}",
                        slurm_info);
        return std::nullopt;
      }

      std::size_t pos = logfile_path.find("%J");
      if (pos != std::string::npos) {
        logfile_path.replace(pos, 2, jobid);
      } else {
        logger()->error("Log file path is in unexpected format! {}", logfile_path);
        return std::nullopt;
      }

      auto status_func = [&](const std::string& job_id) {
        return is_subprocess_running(job_id);
      };

      auto cancel_func = [&](const std::string& job_id) {
        cancel_subprocess(job_id);
      };

      JobRegistry::register_job(managed_task_name,
                                jobid,
                                logfile_path,
                                status_func,
                                cancel_func);

      return std::make_optional(jobid);
    }

    return std::nullopt;
  }

  std::string SlurmLauncher::prepare_launch_cmd(const JobStep& job, bool is_daq2) {
    //if (job.parameters.executable_subdir == "launch_")
    std::string executable =
      job.parameters.lute_location + "/" + job.parameters.executable_subdir
      + "/submit_slurm";

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
      logger()->error("Trying to get information for an empty SLURM jobid! Skipping!");

      return;
    }

    std::string logfile_path = JobRegistry::get_log_file_for(jobid);

    std::string get_slurm_log_cmd { "cat " + logfile_path };
    logger()->debug("Attempting to grab output from log file using: {}",
                    get_slurm_log_cmd);

    // Wait some milliseconds if it fails. Then wait longer until more than 10 s
    int ret_code { 0 };
    std::size_t wait_ms { 500 };
    std::tie(log, ret_code) = run_subprocess_log(get_slurm_log_cmd, true);
    if (WEXITSTATUS(ret_code) == 0) {
      return;
    }
    while (WEXITSTATUS(ret_code) && wait_ms < 10000) {
      logger()->error("Trying to `cat` log file failed. Will wait {} ms and try again.",
                      wait_ms);

      std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
      std::tie(log, ret_code) = run_subprocess_log(get_slurm_log_cmd, true);
      if (WEXITSTATUS(ret_code) == 0) {
        return;
      }
      wait_ms <<= 2;
    }

    logger()->error("Unable to read log file: {}", logfile_path);
  }

  bool PythonLauncher::is_subprocess_running(const std::string& jobid) {
    try {
#if defined(__linux__) || defined(__APPLE__) || defined(__OSX__) || defined(__APPLE_CC__)
      pid_t pid { static_cast<pid_t>(std::stol(jobid)) };

      if (pid <= 0) {
        return false;
      }

      if (kill(pid, 0) == 0) {
        return true;
      }

      const int kill_err { errno };

      return kill_err != ESRCH;
#else
      DWORD pid { static_cast<DWORD>(std::stoul(jobid)) };

      if (pid == 0) {
        return false;
      }

      HANDLE h { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid) };
      if (!h) {
        return false;
      }

      DWORD code;
      bool running { GetExitCodeProcess(h, &code) && code == STILL_ACTIVE };

      CloseHandle(h);

      return running;
#endif
    } catch (const std::exception& e) {
      logger()->error("Likely invalid job ID '{}' passed to is_subprocess_running, got error: {}",
                      jobid,
                      e.what());

      return false;
    }
  }


  void PythonLauncher::cancel_subprocess(const std::string &jobid) {
    // Will use kill -9 <jobid> for the cancel operation
    std::string cancel_cmd { "kill -9 " + jobid };

    // Since it will maybe be called from the JobRegistry, and it could be from
    // inside a signal handler/during rapid teardown, use std::system to just
    // fire it off.

    int ret { std::system(cancel_cmd.c_str()) };
    (void)ret;
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
