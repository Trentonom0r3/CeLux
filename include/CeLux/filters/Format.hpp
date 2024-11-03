#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Format : public FilterBase {
public:
    /**
     * Convert the input video to one of the specified pixel formats.
     */
    /**
     * A '|'-separated list of pixel formats
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setPix_fmts(const std::vector<std::string>& value);
    std::vector<std::string> getPix_fmts() const;

    /**
     * A '|'-separated list of color spaces
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setColor_spaces(const std::string& value);
    std::string getColor_spaces() const;

    /**
     * A '|'-separated list of color ranges
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setColor_ranges(const std::string& value);
    std::string getColor_ranges() const;

    Format(std::vector<std::string> pix_fmts = std::vector<std::string>(), const std::string& color_spaces = "", const std::string& color_ranges = "");
    virtual ~Format();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::vector<std::string> pix_fmts_;
    std::string color_spaces_;
    std::string color_ranges_;
};
