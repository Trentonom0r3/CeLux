#ifndef FFEXCEPTION_HPP
#define FFEXCEPTION_HPP

#include "FFCore.hpp"

#define FF_CHECK(func) do { \
    int errorCode = func; \
    if (errorCode < 0) { \
        throw ffmpy::error::FFException(errorCode); \
    } \
} while (0)
 
#define FF_CHECK_MSG(func, msg) do { \
	int errorCode = func; \
	if (errorCode < 0) { \
		throw ffmpy::error::FFException(msg + ": " +                               \
                                            ffmpy::errorToString(errorCode)); \
	} \
} while (0)


namespace ffmpy {
    namespace error {
        class FFException : public std::exception {
        public:
            explicit FFException(const std::string& message) : errorMessage(message) {}

            // Use FFmpeg::errorToString to be explicit
            explicit FFException(int errorCode) : errorMessage(ffmpy::errorToString(errorCode)) {}

            virtual const char* what() const noexcept override {
                return errorMessage.c_str();
            }

        private:
            std::string errorMessage;
        };
    }
}

#endif // FFEXCEPTION_HPP
