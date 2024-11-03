#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Addroi : public FilterBase {
public:
    /**
     * Add region of interest to frame.
     */
    /**
     * Region distance from left edge of frame.
     * Type: String
     * Required: No
     * Default: 0
     */
    void setRegionDistanceFromLeftEdgeOfFrame(const std::string& value);
    std::string getRegionDistanceFromLeftEdgeOfFrame() const;

    /**
     * Region distance from top edge of frame.
     * Type: String
     * Required: No
     * Default: 0
     */
    void setRegionDistanceFromTopEdgeOfFrame(const std::string& value);
    std::string getRegionDistanceFromTopEdgeOfFrame() const;

    /**
     * Region width.
     * Type: String
     * Required: No
     * Default: 0
     */
    void setRegionWidth(const std::string& value);
    std::string getRegionWidth() const;

    /**
     * Region height.
     * Type: String
     * Required: No
     * Default: 0
     */
    void setRegionHeight(const std::string& value);
    std::string getRegionHeight() const;

    /**
     * Quantisation offset to apply in the region.
     * Type: Rational
     * Required: No
     * Default: 1.59315
     */
    void setQoffset(const std::pair<int, int>& value);
    std::pair<int, int> getQoffset() const;

    /**
     * Remove any existing regions of interest before adding the new one.
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setClear(bool value);
    bool getClear() const;

    Addroi(const std::string& regionDistanceFromLeftEdgeOfFrame = "0", const std::string& regionDistanceFromTopEdgeOfFrame = "0", const std::string& regionWidth = "0", const std::string& regionHeight = "0", std::pair<int, int> qoffset = std::make_pair<int, int>(0, 1), bool clear = false);
    virtual ~Addroi();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string regionDistanceFromLeftEdgeOfFrame_;
    std::string regionDistanceFromTopEdgeOfFrame_;
    std::string regionWidth_;
    std::string regionHeight_;
    std::pair<int, int> qoffset_;
    bool clear_;
};
