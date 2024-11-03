#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Loop : public FilterBase {
public:
    /**
     * Loop video frames.
     */
    /**
     * number of loops
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setLoop(int value);
    int getLoop() const;

    /**
     * max number of frames to loop
     * Type: Integer64
     * Required: No
     * Default: 0
     */
    void setSize(int64_t value);
    int64_t getSize() const;

    /**
     * set the loop start frame
     * Type: Integer64
     * Required: No
     * Default: 0
     */
    void setStart(int64_t value);
    int64_t getStart() const;

    /**
     * set the loop start time
     * Type: Duration
     * Required: No
     * Default: 9223372036854775807
     */
    void setTime(int64_t value);
    int64_t getTime() const;

    Loop(int loop = 0, int64_t size = 0ULL, int64_t start = 0ULL, int64_t time = 9223372036854775807ULL);
    virtual ~Loop();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int loop_;
    int64_t size_;
    int64_t start_;
    int64_t time_;
};
