#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Segment : public FilterBase {
public:
    /**
     * Segment video stream.
     */
    /**
     * timestamps of input at which to split input
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setTimestamps(const std::string& value);
    std::string getTimestamps() const;

    /**
     * frames at which to split input
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFrames(const std::string& value);
    std::string getFrames() const;

    Segment(const std::string& timestamps = "", const std::string& frames = "");
    virtual ~Segment();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string timestamps_;
    std::string frames_;
};
