#include <iostream>
#include <mutex>
#include <string>

class Logger
{
  public:
    enum class Level
    {
        DEBUG = 0,
        INFO,
        WARN,
        ERROR,
        NONE // No logging
    };

    static Logger& getInstance()
    {
        static Logger instance; // Guaranteed to be thread-safe
        return instance;
    }

    // Function to set the global logging level
    void setLogLevel(Level level)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        logLevel = level;
    }

    // Logging functions for different levels
    void logDebug(const std::string& message)
    {
        log(Level::DEBUG, "DEBUG", message);
    }

    void logInfo(const std::string& message)
    {
        log(Level::INFO, "INFO", message);
    }

    void logWarn(const std::string& message)
    {
        log(Level::WARN, "WARN", message);
    }

    void logError(const std::string& message)
    {
        log(Level::ERROR, "ERROR", message);
    }

  private:
    Level logLevel = Level::INFO;
    std::mutex mutex_;

    // Private constructor for singleton pattern
    Logger()
    {
    }

    // Core logging function
    void log(Level level, const std::string& levelStr, const std::string& message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level >= logLevel)
        {
            std::cout << "[" << levelStr << "] " << message << std::endl;
        }
    }
};

// Global function to set logging level
void setGlobalLogLevel(Logger::Level level)
{
    Logger::getInstance().setLogLevel(level);
}


//define helper macros
#define LOG_DEBUG(message) Logger::getInstance().logDebug(message)
#define LOG_INFO(message) Logger::getInstance().logInfo(message)
#define LOG_WARN(message) Logger::getInstance().logWarn(message)
#define LOG_ERROR(message) Logger::getInstance().logError(message)

