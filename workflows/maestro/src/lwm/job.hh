#ifndef LWM_JOB_HH
#define LWM_JOB_HH

#include <chrono>
#include <concepts>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace LWM {
  enum class TriggerRule {
    ALL_SUCCESS=0, // All previous job steps must exit successfully
    ANY_SUCCESS=1, // Run as soon as any previous job step exits successfully
    ALL_COMPLETED=2, // Run when all previous job steps complete, successful or not

    ALL_FAILED=3,
    ANY_FAILED=4,

    ALWAYS=5
  };

  std::string rule_to_string(TriggerRule& rule);

  class JobParameters {
  public:
    std::string lute_location;
    std::string executable_subdir;
    std::string config_file;
    bool debug;
  };

  /**
   * A unique identifier for a JobStep within a workflow.
   *
   * The JobIdentifier consists of a unique job_key for the JobStep within the DAG,
   * which is generally provided by maestro. Optionally, the identifier tracks a
   * name (e.g. managed Task name) provided by the Executor. There is also a Launcher
   * ID -- this maps the maestro identifiers to physical system resources (process IDs).
   * as well as a name for JobStep (e.g. managed Task name).
   */
  struct JobIdentifier {
    std::string job_key;                  ///< Unique identifier across all DAGs (from maestro)
    std::optional<std::string> name;      ///< A step name from the Executor (no uniqueness guarantee)
    std::string launch_id;                ///< Identifier used by the Launcher to map onto system (pid, etc.)

    inline bool operator<(const JobIdentifier& other) const {
      // Use a full lexical ordering - std::nullopt sorts first in this case.
      return
        std::tie(job_key, name, launch_id) <
        std::tie(other.job_key, other.name, other.launch_id);
    }

    /**
     * Check for partially filled JobIdentifier matches.
     *
     * @note This routine is one-directional! It requires that the stored value have
     *       every field for lookup. Adding fields over time will not match! See the
     *       symmetric implementations below (`compatible`).
     *
     * @param[in] partial A partially filled JobIdentifier.
     * @param[in] full A completely filled JobIdentifier.
     * @returns true if the fully filled `full` matches on every key that is
     *          present for the partially filled `partial`. Otherwise false.
     */
    static bool matches(const JobIdentifier& partial, const JobIdentifier& full);

    /**
     * Construct a vector of iterators that match a partial JobIdentifier.
     *
     * @note The routine currently handles map, and set inputs.
     *
     * @tparam Map The container of identifiers, supporting maps or sets.
     * @tparam Pred A predicate for comparisons, with signature Pred(id, id) -> bool.
     * @param[in] id_map The JobIdentifiers to check for matches against.
     * @param[in] partial The (potentially) partial JobIdentifier to match.
     * @param[in] pred The matching/lookup predicate for field comparisons. Defaults
     *            to JobIdentifier::matches. See notes there on ordering and partial
     *            JobIdentifiers.
     * @returns A vector of iterators to the found matches.
     */
    template <typename Map, typename Pred = decltype(&matches)>
    requires requires(Pred pred, const JobIdentifier& id_a, const JobIdentifier& id_b) {
      { pred(id_a, id_b) } -> std::convertible_to<bool>;
    }
    static inline auto find_matching(Map& id_map,
                                     const JobIdentifier& partial,
                                     Pred&& pred = &matches) {
      using ItType = decltype(id_map.begin());
      using ObjType = std::iter_reference_t<ItType>;

      auto key_for = [](ItType it) -> const JobIdentifier& {
        if constexpr (std::is_same_v<ObjType, const JobIdentifier&>) {
          return *it;
        } else {
          return it->first;
        }
      };

      std::vector<ItType> out;
      if (!partial.job_key.empty()) {
        auto it { id_map.lower_bound(JobIdentifier { partial.job_key }) };
        while (it != id_map.end()) {
          const auto& job_id { key_for(it) };
          if (job_id.job_key != partial.job_key) {
            break;
          } else if (pred(partial, job_id)) {
            out.push_back(it);
          }

          ++it;
        }
      } else {
        for (auto it = id_map.begin(); it != id_map.end(); ++it) {
          if (pred(partial, key_for(it))) {
            out.push_back(it);
          }
        }
      }

      return out;
    }

    /**
     * Check if every set field matches between two JobIdentifiers.
     *
     * @note This doesn't use ordering, and allows either of side of the comparison
     *       to be more complete than the other.
     *
     * @param[in] id_a First JobIdentifier.
     * @param[in] id_b Second JobIdentifier.
     * @returns true if all fields, set in BOTH JobIdentifiers, agree. Otherwise, false.
     */
    static bool compatible(const JobIdentifier& id_a, const JobIdentifier& id_b);
  };

  class JobStep {
  public:
    JobStep(std::string _task_name,
            TriggerRule _trigger_rule,
            JobParameters _parameters,
            std::string _extra_parameters,
            std::vector<JobStep> _next)
      : managed_task_name(_task_name)
      , trigger_rule(_trigger_rule)
      , parameters(_parameters)
      , extra_parameters(_extra_parameters)
      , next(_next)
    {}

    std::string managed_task_name;
    TriggerRule trigger_rule;
    JobParameters parameters;
    std::string extra_parameters;
    std::vector<JobStep> next;
  };

  /**
   * A simple struct that holds the timing information for an individual
   * JobStep. The time the Launcher submits the step, the time the JobStep is
   * confirmed to be running, and the time the JobStep exits are recorded.
   */
  class JobStepSplits {
  public:
    std::chrono::time_point<std::chrono::steady_clock>
        launch_point; ///< Time Launcher submits the JobStep
    std::chrono::time_point<std::chrono::steady_clock>
        running_point; ///< Time JobStep reports it is running
    std::chrono::time_point<std::chrono::steady_clock>
        end_point; ///< Time JobStep completes
  };

  struct LaunchContext {
    std::string workflow_id {};                    ///< Identifier, for multi-DAG management
    std::map<std::string, std::string> env {};     ///< Any required environment variables (EXPERIMENT, RUN_NUM)
    std::vector<std::string> extra_launch_args {}; ///< Additional launch args (--psana2)
    std::map<std::string, std::string> labels {};  ///< Space for additional unit labels
  };

  class JobReturn {
  public:
    JobReturn(std::string _name,
              std::string _status,
              std::string _log,
              JobStepSplits _splits)
      : managed_task_name(_name)
      , status(_status)
      , log(_log)
      , splits(_splits)
    {}

    // Specify move since it may be returned from function
    JobReturn(JobReturn&& other) noexcept
      : managed_task_name(std::move(other.managed_task_name))
      , status(std::move(other.status))
      , log(std::move(other.log))
      , splits(std::move(other.splits))
    {}
    std::string managed_task_name;
    std::string status;
    std::string log;
    JobStepSplits splits;
  };

  class JobStepInfo {
  public:
    JobStep job;      ///< The job
    std::string host; ///< Host this job is running on/ran on
  };
};
#endif
