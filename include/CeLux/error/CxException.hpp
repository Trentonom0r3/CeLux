#ifndef CxException_HPP
#define CxException_HPP

#include "CxCore.hpp"

#define FF_CHECK(func) do { \
    int errorCode = func; \
    if (errorCode < 0) { \
        throw celux::error::CxException(errorCode); \
    } \
} while (0)
 
#define FF_CHECK_MSG(func, msg) do { \
	int errorCode = func; \
	if (errorCode < 0) { \
		throw celux::error::CxException(msg + ": " +                               \
                                            celux::errorToString(errorCode)); \
	} \
} while (0)


namespace celux {
    namespace error {
        class CxException : public std::exception {
        public:
            explicit CxException(const std::string& message) : errorMessage(message) {}

            // Use FFmpeg::errorToString to be explicit
            explicit CxException(int errorCode) : errorMessage(celux::errorToString(errorCode)) {}

            virtual const char* what() const noexcept override {
                return errorMessage.c_str();
            }

        private:
            std::string errorMessage;
        };
    }
}

#endif // CxException_HPP
