#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ciescope : public FilterBase {
public:
    /**
     * Video CIE scope.
     */
    /**
     * set color system
     * Unit: system
     * Possible Values: ntsc (0), 470m (0), ebu (1), 470bg (1), smpte (2), 240m (3), apple (4), widergb (5), cie1931 (6), hdtv (7), rec709 (7), uhdtv (8), rec2020 (8), dcip3 (9)
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setSystem(int value);
    int getSystem() const;

    /**
     * set cie system
     * Unit: cie
     * Possible Values: xyy (0), ucs (1), luv (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCie(int value);
    int getCie() const;

    /**
     * set what gamuts to draw
     * Unit: gamuts
     * Possible Values: ntsc (1), 470m (1), ebu (2), 470bg (2), smpte (4), 240m (8), apple (16), widergb (32), cie1931 (64), hdtv (128), rec709 (128), uhdtv (256), rec2020 (256), dcip3 (512)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setGamuts(int value);
    int getGamuts() const;

    /**
     * set ciescope size
     * Aliases: s
     * Type: Integer
     * Required: No
     * Default: 512
     */
    void setSize(int value);
    int getSize() const;

    /**
     * set ciescope intensity
     * Aliases: i
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIntensity(float value);
    float getIntensity() const;

    /**
     * 
     * Type: Float
     * Required: No
     * Default: 0.75
     */
    void setContrast(float value);
    float getContrast() const;

    /**
     * 
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setCorrgamma(bool value);
    bool getCorrgamma() const;

    /**
     * 
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setShowwhite(bool value);
    bool getShowwhite() const;

    /**
     * 
     * Type: Double
     * Required: No
     * Default: 2.60
     */
    void setGamma(double value);
    double getGamma() const;

    /**
     * fill with CIE colors
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setFill(bool value);
    bool getFill() const;

    Ciescope(int system = 7, int cie = 0, int gamuts = 0, int size = 512, float intensity = 0.00, float contrast = 0.75, bool corrgamma = true, bool showwhite = false, double gamma = 2.60, bool fill = true);
    virtual ~Ciescope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int system_;
    int cie_;
    int gamuts_;
    int size_;
    float intensity_;
    float contrast_;
    bool corrgamma_;
    bool showwhite_;
    double gamma_;
    bool fill_;
};
