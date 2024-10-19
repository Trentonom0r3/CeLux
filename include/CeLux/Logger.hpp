// src/logger.h

#ifndef CELEX_LOGGER_H
#define CELEX_LOGGER_H

#include <memory>
#include <spdlog/spdlog.h>

namespace celux
{

class Logger
{
  public:
    // Retrieves the singleton instance
    static std::shared_ptr<spdlog::logger>& get_logger();

    // Configures the logger's verbosity
    static void set_level(spdlog::level::level_enum level);

  private:
    Logger() = default;
    ~Logger() = default;

    // Deleted to prevent copying
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static std::shared_ptr<spdlog::logger> logger_instance;
};

} // namespace celux


//conveniece macros
#define CELUX_TRACE(...) celux::Logger::get_logger()->trace(__VA_ARGS__)
#define CELUX_DEBUG(...) celux::Logger::get_logger()->debug(__VA_ARGS__)
#define CELUX_INFO(...) celux::Logger::get_logger()->info(__VA_ARGS__)
#define CELUX_WARN(...) celux::Logger::get_logger()->warn(__VA_ARGS__)
#define CELUX_ERROR(...) celux::Logger::get_logger()->error(__VA_ARGS__)
#define CELUX_CRITICAL(...) celux::Logger::get_logger()->critical(__VA_ARGS__)


#endif // CELEX_LOGGER_H
