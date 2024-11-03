#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Maskedmerge : public FilterBase {
public:
    /**
     * Merge first stream with second stream using third stream as mask.
     */
    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Maskedmerge(int planes = 15);
    virtual ~Maskedmerge();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
};
