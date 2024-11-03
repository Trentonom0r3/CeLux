#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Freezeframes : public FilterBase {
public:
    /**
     * Freeze video frames.
     */
    /**
     * set first frame to freeze
     * Type: Integer64
     * Required: No
     * Default: 0
     */
    void setFirst(int64_t value);
    int64_t getFirst() const;

    /**
     * set last frame to freeze
     * Type: Integer64
     * Required: No
     * Default: 0
     */
    void setLast(int64_t value);
    int64_t getLast() const;

    /**
     * set frame to replace
     * Type: Integer64
     * Required: No
     * Default: 0
     */
    void setReplace(int64_t value);
    int64_t getReplace() const;

    Freezeframes(int64_t first = 0ULL, int64_t last = 0ULL, int64_t replace = 0ULL);
    virtual ~Freezeframes();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int64_t first_;
    int64_t last_;
    int64_t replace_;
};
