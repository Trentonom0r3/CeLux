#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Buffersink : public FilterBase {
public:
    /**
     * Buffer video frames, and make them available to the end of the filter graph.
     */
    /**
     * set the supported pixel formats
     * Type: Binary
     * Required: Yes
     * Default: No Default
     */
    void setPix_fmts(const std::vector<std::string>& value);
    std::vector<std::string> getPix_fmts() const;

    /**
     * set the supported color spaces
     * Type: Binary
     * Required: Yes
     * Default: No Default
     */
    void setColor_spaces(const std::vector<uint8_t>& value);
    std::vector<uint8_t> getColor_spaces() const;

    /**
     * set the supported color ranges
     * Type: Binary
     * Required: Yes
     * Default: No Default
     */
    void setColor_ranges(const std::vector<uint8_t>& value);
    std::vector<uint8_t> getColor_ranges() const;

    Buffersink(std::vector<std::string> pix_fmts = std::vector<std::string>(), std::vector<uint8_t> color_spaces = std::vector<uint8_t>(), std::vector<uint8_t> color_ranges = std::vector<uint8_t>());
    virtual ~Buffersink();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::vector<std::string> pix_fmts_;
    std::vector<uint8_t> color_spaces_;
    std::vector<uint8_t> color_ranges_;
};
