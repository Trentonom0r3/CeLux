#ifndef FFEXCEPTION_HPP
#define FFEXCEPTION_HPP

#include <exception>
#include <string>
#include "FFCore.hpp"

#define FF_CHECK(func) do { \
    int errorCode = func; \
    if (errorCode < 0) { \
        throw FFmpeg::Error::FFException(errorCode); \
    } \
} while (0)

namespace FFmpeg {
    namespace Error {
        class FFException : public std::exception {
        public:
            explicit FFException(const std::string& message) : errorMessage(message) {}

            // Use FFmpeg::errorToString to be explicit
            explicit FFException(int errorCode) : errorMessage(FFmpeg::errorToString(errorCode)) {}

            virtual const char* what() const noexcept override {
                return errorMessage.c_str();
            }

        private:
            std::string errorMessage;
        };
    }
}

#endif // FFEXCEPTION_HPP
