#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showcqt : public FilterBase {
public:
    /**
     * Convert input audio to a CQT (Constant/Clamped Q Transform) spectrum video output.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 1920x1080
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set video rate
     * Aliases: r, fps
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set bargraph height
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setBar_h(int value);
    int getBar_h() const;

    /**
     * set axis height
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setAxis_h(int value);
    int getAxis_h() const;

    /**
     * set sonogram height
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setSono_h(int value);
    int getSono_h() const;

    /**
     * set fullhd size
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setFullhd(bool value);
    bool getFullhd() const;

    /**
     * set sonogram volume
     * Aliases: sono_v
     * Type: String
     * Required: No
     * Default: 16
     */
    void setVolume(const std::string& value);
    std::string getVolume() const;

    /**
     * set bargraph volume
     * Aliases: bar_v
     * Type: String
     * Required: No
     * Default: sono_v
     */
    void setVolume2(const std::string& value);
    std::string getVolume2() const;

    /**
     * set sonogram gamma
     * Aliases: gamma
     * Type: Float
     * Required: No
     * Default: 3.00
     */
    void setSono_g(float value);
    float getSono_g() const;

    /**
     * set bargraph gamma
     * Aliases: bar_g
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setGamma2(float value);
    float getGamma2() const;

    /**
     * set bar transparency
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setBar_t(float value);
    float getBar_t() const;

    /**
     * set timeclamp
     * Aliases: tc
     * Type: Double
     * Required: No
     * Default: 0.17
     */
    void setTimeclamp(double value);
    double getTimeclamp() const;

    /**
     * set attack time
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setAttack(double value);
    double getAttack() const;

    /**
     * set base frequency
     * Type: Double
     * Required: No
     * Default: 20.02
     */
    void setBasefreq(double value);
    double getBasefreq() const;

    /**
     * set end frequency
     * Type: Double
     * Required: No
     * Default: 20495.60
     */
    void setEndfreq(double value);
    double getEndfreq() const;

    /**
     * set coeffclamp
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setCoeffclamp(float value);
    float getCoeffclamp() const;

    /**
     * set tlength
     * Type: String
     * Required: No
     * Default: 384*tc/(384+tc*f)
     */
    void setTlength(const std::string& value);
    std::string getTlength() const;

    /**
     * set transform count
     * Type: Integer
     * Required: No
     * Default: 6
     */
    void setCount(int value);
    int getCount() const;

    /**
     * set frequency count
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFcount(int value);
    int getFcount() const;

    /**
     * set axis font file
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFontfile(const std::string& value);
    std::string getFontfile() const;

    /**
     * set axis font
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFont(const std::string& value);
    std::string getFont() const;

    /**
     * set font color
     * Type: String
     * Required: No
     * Default: st(0, (midi(f)-59.5)/12);st(1, if(between(ld(0),0,1), 0.5-0.5*cos(2*PI*ld(0)), 0));r(1-ld(1)) + b(ld(1))
     */
    void setFontcolor(const std::string& value);
    std::string getFontcolor() const;

    /**
     * set axis image
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setAxisfile(const std::string& value);
    std::string getAxisfile() const;

    /**
     * draw axis
     * Aliases: axis
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setText(bool value);
    bool getText() const;

    /**
     * set color space
     * Unit: csp
     * Possible Values: unspecified (2), bt709 (1), fcc (4), bt470bg (5), smpte170m (6), smpte240m (7), bt2020ncl (9)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setCsp(int value);
    int getCsp() const;

    /**
     * set color scheme
     * Type: String
     * Required: No
     * Default: 1|0.5|0|0|0.5|1
     */
    void setCscheme(const std::string& value);
    std::string getCscheme() const;

    Showcqt(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int bar_h = -1, int axis_h = -1, int sono_h = -1, bool fullhd = true, const std::string& volume = "16", const std::string& volume2 = "sono_v", float sono_g = 3.00, float gamma2 = 1.00, float bar_t = 1.00, double timeclamp = 0.17, double attack = 0.00, double basefreq = 20.02, double endfreq = 20495.60, float coeffclamp = 1.00, const std::string& tlength = "384*tc/(384+tc*f)", int count = 6, int fcount = 0, const std::string& fontfile = "", const std::string& font = "", const std::string& fontcolor = "st(0, (midi(f)-59.5)/12);st(1, if(between(ld(0),0,1), 0.5-0.5*cos(2*PI*ld(0)), 0));r(1-ld(1)) + b(ld(1))", const std::string& axisfile = "", bool text = true, int csp = 2, const std::string& cscheme = "1|0.5|0|0|0.5|1");
    virtual ~Showcqt();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    int bar_h_;
    int axis_h_;
    int sono_h_;
    bool fullhd_;
    std::string volume_;
    std::string volume2_;
    float sono_g_;
    float gamma2_;
    float bar_t_;
    double timeclamp_;
    double attack_;
    double basefreq_;
    double endfreq_;
    float coeffclamp_;
    std::string tlength_;
    int count_;
    int fcount_;
    std::string fontfile_;
    std::string font_;
    std::string fontcolor_;
    std::string axisfile_;
    bool text_;
    int csp_;
    std::string cscheme_;
};
