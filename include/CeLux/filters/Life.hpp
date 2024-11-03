#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Life : public FilterBase {
public:
    /**
     * Create life.
     */
    /**
     * set source file
     * Aliases: f
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFilename(const std::string& value);
    std::string getFilename() const;

    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: Yes
     * Default: No Default
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

    /**
     * set rule
     * Type: String
     * Required: No
     * Default: B3/S23
     */
    void setRule(const std::string& value);
    std::string getRule() const;

    /**
     * set fill ratio for filling initial grid randomly
     * Aliases: ratio
     * Type: Double
     * Required: No
     * Default: 0.62
     */
    void setRandom_fill_ratio(double value);
    double getRandom_fill_ratio() const;

    /**
     * set the seed for filling the initial grid randomly
     * Aliases: seed
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setRandom_seed(int64_t value);
    int64_t getRandom_seed() const;

    /**
     * stitch boundaries
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setStitch(bool value);
    bool getStitch() const;

    /**
     * set mold speed for dead cells
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMold(int value);
    int getMold() const;

    /**
     * set life color
     * Type: Color
     * Required: No
     * Default: white
     */
    void setLife_color(const std::string& value);
    std::string getLife_color() const;

    /**
     * set death color
     * Type: Color
     * Required: No
     * Default: black
     */
    void setDeath_color(const std::string& value);
    std::string getDeath_color() const;

    /**
     * set mold color
     * Type: Color
     * Required: No
     * Default: black
     */
    void setMold_color(const std::string& value);
    std::string getMold_color() const;

    Life(const std::string& filename = "", std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), const std::string& rule = "B3/S23", double random_fill_ratio = 0.62, int64_t random_seed = 0, bool stitch = true, int mold = 0, const std::string& life_color = "white", const std::string& death_color = "black", const std::string& mold_color = "black");
    virtual ~Life();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string filename_;
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    std::string rule_;
    double random_fill_ratio_;
    int64_t random_seed_;
    bool stitch_;
    int mold_;
    std::string life_color_;
    std::string death_color_;
    std::string mold_color_;
};
