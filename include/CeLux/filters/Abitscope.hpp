#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Abitscope : public FilterBase {
public:
    /**
     * Convert input audio to audio bit scope video output.
     */
    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 1024x256
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set channels colors
     * Type: String
     * Required: No
     * Default: red|green|blue|yellow|orange|lime|pink|magenta|brown
     */
    void setColors(const std::string& value);
    std::string getColors() const;

    /**
     * set output mode
     * Aliases: m
     * Unit: mode
     * Possible Values: bars (0), trace (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    Abitscope(std::pair<int, int> rate = std::make_pair<int, int>(0, 1), std::pair<int, int> size = std::make_pair<int, int>(0, 1), const std::string& colors = "red|green|blue|yellow|orange|lime|pink|magenta|brown", int mode = 0);
    virtual ~Abitscope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> rate_;
    std::pair<int, int> size_;
    std::string colors_;
    int mode_;
};
