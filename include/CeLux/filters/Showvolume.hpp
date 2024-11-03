#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showvolume : public FilterBase {
public:
    /**
     * Convert input audio volume to video output.
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
     * set border width
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setBorderWidth(int value);
    int getBorderWidth() const;

    /**
     * set channel width
     * Type: Integer
     * Required: No
     * Default: 400
     */
    void setChannelWidth(int value);
    int getChannelWidth() const;

    /**
     * set channel height
     * Type: Integer
     * Required: No
     * Default: 20
     */
    void setChannelHeight(int value);
    int getChannelHeight() const;

    /**
     * set fade
     * Type: Double
     * Required: No
     * Default: 0.95
     */
    void setFade(double value);
    double getFade() const;

    /**
     * set volume color expression
     * Type: String
     * Required: No
     * Default: PEAK*255+floor((1-PEAK)*255)*256+0xff000000
     */
    void setVolumeColor(const std::string& value);
    std::string getVolumeColor() const;

    /**
     * display channel names
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setDisplayChannelNames(bool value);
    bool getDisplayChannelNames() const;

    /**
     * display volume value
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setDisplayVolume(bool value);
    bool getDisplayVolume() const;

    /**
     * duration for max value display
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setDm(double value);
    double getDm() const;

    /**
     * set color of the max value line
     * Type: Color
     * Required: No
     * Default: orange
     */
    void setDmc(const std::string& value);
    std::string getDmc() const;

    /**
     * set orientation
     * Unit: orientation
     * Possible Values: h (0), v (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOrientation(int value);
    int getOrientation() const;

    /**
     * set step size
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStepSize(int value);
    int getStepSize() const;

    /**
     * set background opacity
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBackgroundOpacity(float value);
    float getBackgroundOpacity() const;

    /**
     * set mode
     * Unit: mode
     * Possible Values: p (0), r (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set display scale
     * Unit: display_scale
     * Possible Values: lin (0), log (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDs(int value);
    int getDs() const;

    Showvolume(std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int borderWidth = 1, int channelWidth = 400, int channelHeight = 20, double fade = 0.95, const std::string& volumeColor = "PEAK*255+floor((1-PEAK)*255)*256+0xff000000", bool displayChannelNames = true, bool displayVolume = true, double dm = 0.00, const std::string& dmc = "orange", int orientation = 0, int stepSize = 0, float backgroundOpacity = 0.00, int mode = 0, int ds = 0);
    virtual ~Showvolume();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> rate_;
    int borderWidth_;
    int channelWidth_;
    int channelHeight_;
    double fade_;
    std::string volumeColor_;
    bool displayChannelNames_;
    bool displayVolume_;
    double dm_;
    std::string dmc_;
    int orientation_;
    int stepSize_;
    float backgroundOpacity_;
    int mode_;
    int ds_;
};
