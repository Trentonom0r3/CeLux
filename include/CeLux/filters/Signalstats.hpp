#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Signalstats : public FilterBase {
public:
    /**
     * Generate statistics from video analysis.
     */
    /**
     * set statistics filters
     * Unit: filters
     * Possible Values: tout (1), vrep (2), brng (4)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setStat(int value);
    int getStat() const;

    /**
     * set video filter
     * Unit: out
     * Possible Values: tout (0), vrep (1), brng (2)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setOut(int value);
    int getOut() const;

    /**
     * set highlight color
     * Aliases: c
     * Type: Color
     * Required: No
     * Default: yellow
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    Signalstats(int stat = 0, int out = -1, const std::string& color = "yellow");
    virtual ~Signalstats();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int stat_;
    int out_;
    std::string color_;
};
