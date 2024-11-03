#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Shuffleframes : public FilterBase {
public:
    /**
     * Shuffle video frames.
     */
    /**
     * set destination indexes of input frames
     * Type: String
     * Required: No
     * Default: 0
     */
    void setMapping(const std::string& value);
    std::string getMapping() const;

    Shuffleframes(const std::string& mapping = "0");
    virtual ~Shuffleframes();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string mapping_;
};
