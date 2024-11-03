#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Adrawgraph : public FilterBase {
public:
    /**
     * Draw a graph using input audio metadata.
     */
    /**
     * set 1st metadata key
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setM1(const std::string& value);
    std::string getM1() const;

    /**
     * set 1st foreground color expression
     * Type: String
     * Required: No
     * Default: 0xffff0000
     */
    void setFg1(const std::string& value);
    std::string getFg1() const;

    /**
     * set 2nd metadata key
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setM2(const std::string& value);
    std::string getM2() const;

    /**
     * set 2nd foreground color expression
     * Type: String
     * Required: No
     * Default: 0xff00ff00
     */
    void setFg2(const std::string& value);
    std::string getFg2() const;

    /**
     * set 3rd metadata key
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setM3(const std::string& value);
    std::string getM3() const;

    /**
     * set 3rd foreground color expression
     * Type: String
     * Required: No
     * Default: 0xffff00ff
     */
    void setFg3(const std::string& value);
    std::string getFg3() const;

    /**
     * set 4th metadata key
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setM4(const std::string& value);
    std::string getM4() const;

    /**
     * set 4th foreground color expression
     * Type: String
     * Required: No
     * Default: 0xffffff00
     */
    void setFg4(const std::string& value);
    std::string getFg4() const;

    /**
     * set background color
     * Type: Color
     * Required: No
     * Default: white
     */
    void setBg(const std::string& value);
    std::string getBg() const;

    /**
     * set minimal value
     * Type: Float
     * Required: No
     * Default: -1.00
     */
    void setMin(float value);
    float getMin() const;

    /**
     * set maximal value
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setMax(float value);
    float getMax() const;

    /**
     * set graph mode
     * Unit: mode
     * Possible Values: bar (0), dot (1), line (2)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set slide mode
     * Unit: slide
     * Possible Values: frame (0), replace (1), scroll (2), rscroll (3), picture (4)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSlide(int value);
    int getSlide() const;

    /**
     * set graph size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 900x256
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    Adrawgraph(const std::string& m1 = "", const std::string& fg1 = "0xffff0000", const std::string& m2 = "", const std::string& fg2 = "0xff00ff00", const std::string& m3 = "", const std::string& fg3 = "0xffff00ff", const std::string& m4 = "", const std::string& fg4 = "0xffffff00", const std::string& bg = "white", float min = -1.00, float max = 1.00, int mode = 2, int slide = 0, std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1));
    virtual ~Adrawgraph();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string m1_;
    std::string fg1_;
    std::string m2_;
    std::string fg2_;
    std::string m3_;
    std::string fg3_;
    std::string m4_;
    std::string fg4_;
    std::string bg_;
    float min_;
    float max_;
    int mode_;
    int slide_;
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
};
