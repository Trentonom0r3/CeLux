#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Sierpinski : public FilterBase {
public:
    /**
     * Render a Sierpinski fractal.
     */
    /**
     * set frame size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 640x480
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set frame rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set the seed
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setSeed(int64_t value);
    int64_t getSeed() const;

    /**
     * set the jump
     * Type: Integer
     * Required: No
     * Default: 100
     */
    void setJump(int value);
    int getJump() const;

    /**
     * set fractal type
     * Unit: type
     * Possible Values: carpet (0), triangle (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setType(int value);
    int getType() const;

    Sierpinski(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int64_t seed = 0, int jump = 100, int type = 0);
    virtual ~Sierpinski();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    int64_t seed_;
    int jump_;
    int type_;
};
