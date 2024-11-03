#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Avectorscope : public FilterBase {
public:
    /**
     * Convert input audio to vectorscope video output.
     */
    /**
     * set mode
     * Aliases: m
     * Unit: mode
     * Possible Values: lissajous (0), lissajous_xy (1), polar (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

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
     * Default: 400x400
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set red contrast
     * Type: Integer
     * Required: No
     * Default: 40
     */
    void setRc(int value);
    int getRc() const;

    /**
     * set green contrast
     * Type: Integer
     * Required: No
     * Default: 160
     */
    void setGc(int value);
    int getGc() const;

    /**
     * set blue contrast
     * Type: Integer
     * Required: No
     * Default: 80
     */
    void setBc(int value);
    int getBc() const;

    /**
     * set alpha contrast
     * Type: Integer
     * Required: No
     * Default: 255
     */
    void setAc(int value);
    int getAc() const;

    /**
     * set red fade
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setRf(int value);
    int getRf() const;

    /**
     * set green fade
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setGf(int value);
    int getGf() const;

    /**
     * set blue fade
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setBf(int value);
    int getBf() const;

    /**
     * set alpha fade
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setAf(int value);
    int getAf() const;

    /**
     * set zoom factor
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setZoom(double value);
    double getZoom() const;

    /**
     * set draw mode
     * Unit: draw
     * Possible Values: dot (0), line (1), aaline (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDraw(int value);
    int getDraw() const;

    /**
     * set amplitude scale mode
     * Unit: scale
     * Possible Values: lin (0), sqrt (1), cbrt (2), log (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setScale(int value);
    int getScale() const;

    /**
     * swap x axis with y axis
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setSwap(bool value);
    bool getSwap() const;

    /**
     * mirror axis
     * Unit: mirror
     * Possible Values: none (0), x (1), y (2), xy (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMirror(int value);
    int getMirror() const;

    Avectorscope(int mode = 0, std::pair<int, int> rate = std::make_pair<int, int>(0, 1), std::pair<int, int> size = std::make_pair<int, int>(0, 1), int rc = 40, int gc = 160, int bc = 80, int ac = 255, int rf = 15, int gf = 10, int bf = 5, int af = 5, double zoom = 1.00, int draw = 0, int scale = 0, bool swap = true, int mirror = 0);
    virtual ~Avectorscope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    std::pair<int, int> rate_;
    std::pair<int, int> size_;
    int rc_;
    int gc_;
    int bc_;
    int ac_;
    int rf_;
    int gf_;
    int bf_;
    int af_;
    double zoom_;
    int draw_;
    int scale_;
    bool swap_;
    int mirror_;
};
