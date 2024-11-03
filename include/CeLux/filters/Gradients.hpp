#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Gradients : public FilterBase {
public:
    /**
     * Draw a gradients.
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
     * set 1st color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC0(const std::string& value);
    std::string getC0() const;

    /**
     * set 2nd color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC1(const std::string& value);
    std::string getC1() const;

    /**
     * set 3rd color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC2(const std::string& value);
    std::string getC2() const;

    /**
     * set 4th color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC3(const std::string& value);
    std::string getC3() const;

    /**
     * set 5th color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC4(const std::string& value);
    std::string getC4() const;

    /**
     * set 6th color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC5(const std::string& value);
    std::string getC5() const;

    /**
     * set 7th color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC6(const std::string& value);
    std::string getC6() const;

    /**
     * set 8th color
     * Type: Color
     * Required: No
     * Default: random
     */
    void setC7(const std::string& value);
    std::string getC7() const;

    /**
     * set gradient line source x0
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setX0(int value);
    int getX0() const;

    /**
     * set gradient line source y0
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setY0(int value);
    int getY0() const;

    /**
     * set gradient line destination x1
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setX1(int value);
    int getX1() const;

    /**
     * set gradient line destination y1
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setY1(int value);
    int getY1() const;

    /**
     * set the number of colors
     * Aliases: n
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setNb_colors(int value);
    int getNb_colors() const;

    /**
     * set the seed
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setSeed(int64_t value);
    int64_t getSeed() const;

    /**
     * set video duration
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: -1
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    /**
     * set gradients rotation speed
     * Type: Float
     * Required: No
     * Default: 0.01
     */
    void setSpeed(float value);
    float getSpeed() const;

    /**
     * set gradient type
     * Aliases: t
     * Unit: type
     * Possible Values: linear (0), radial (1), circular (2), spiral (3), square (4)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setType(int value);
    int getType() const;

    Gradients(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), const std::string& c0 = "random", const std::string& c1 = "random", const std::string& c2 = "random", const std::string& c3 = "random", const std::string& c4 = "random", const std::string& c5 = "random", const std::string& c6 = "random", const std::string& c7 = "random", int x0 = -1, int y0 = -1, int x1 = -1, int y1 = -1, int nb_colors = 2, int64_t seed = 0, int64_t duration = 0, float speed = 0.01, int type = 0);
    virtual ~Gradients();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    std::string c0_;
    std::string c1_;
    std::string c2_;
    std::string c3_;
    std::string c4_;
    std::string c5_;
    std::string c6_;
    std::string c7_;
    int x0_;
    int y0_;
    int x1_;
    int y1_;
    int nb_colors_;
    int64_t seed_;
    int64_t duration_;
    float speed_;
    int type_;
};
