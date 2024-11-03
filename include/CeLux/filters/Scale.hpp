#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Scale : public FilterBase {
public:
    /**
     * Scale the input video size and/or convert the image format.
     */
    /**
     * Output video width
     * Aliases: w
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setWidth(const std::string& value);
    std::string getWidth() const;

    /**
     * Output video height
     * Aliases: h
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setHeight(const std::string& value);
    std::string getHeight() const;

    /**
     * Flags to pass to libswscale
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFlags(const std::string& value);
    std::string getFlags() const;

    /**
     * set interlacing
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setInterl(bool value);
    bool getInterl() const;

    /**
     * set video size
     * Aliases: s
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setSize(const std::string& value);
    std::string getSize() const;

    /**
     * set input YCbCr type
     * Unit: color
     * Possible Values: auto (-1), bt601 (5), bt470 (5), smpte170m (5), bt709 (1), fcc (4), smpte240m (7), bt2020 (9)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setIn_color_matrix(int value);
    int getIn_color_matrix() const;

    /**
     * set output YCbCr type
     * Unit: color
     * Possible Values: auto (-1), bt601 (5), bt470 (5), smpte170m (5), bt709 (1), fcc (4), smpte240m (7), bt2020 (9)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setOut_color_matrix(int value);
    int getOut_color_matrix() const;

    /**
     * set input color range
     * Unit: range
     * Possible Values: auto (0), unknown (0), full (2), limited (1), jpeg (2), mpeg (1), tv (1), pc (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setIn_range(int value);
    int getIn_range() const;

    /**
     * set output color range
     * Unit: range
     * Possible Values: auto (0), unknown (0), full (2), limited (1), jpeg (2), mpeg (1), tv (1), pc (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOut_range(int value);
    int getOut_range() const;

    /**
     * input vertical chroma position in luma grid/256
     * Type: Integer
     * Required: No
     * Default: -513
     */
    void setIn_v_chr_pos(int value);
    int getIn_v_chr_pos() const;

    /**
     * input horizontal chroma position in luma grid/256
     * Type: Integer
     * Required: No
     * Default: -513
     */
    void setIn_h_chr_pos(int value);
    int getIn_h_chr_pos() const;

    /**
     * output vertical chroma position in luma grid/256
     * Type: Integer
     * Required: No
     * Default: -513
     */
    void setOut_v_chr_pos(int value);
    int getOut_v_chr_pos() const;

    /**
     * output horizontal chroma position in luma grid/256
     * Type: Integer
     * Required: No
     * Default: -513
     */
    void setOut_h_chr_pos(int value);
    int getOut_h_chr_pos() const;

    /**
     * decrease or increase w/h if necessary to keep the original AR
     * Unit: force_oar
     * Possible Values: disable (0), decrease (1), increase (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setForce_original_aspect_ratio(int value);
    int getForce_original_aspect_ratio() const;

    /**
     * enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setForce_divisible_by(int value);
    int getForce_divisible_by() const;

    /**
     * Scaler param 0
     * Type: Double
     * Required: No
     * Default: 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00
     */
    void setParam0(double value);
    double getParam0() const;

    /**
     * Scaler param 1
     * Type: Double
     * Required: No
     * Default: 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00
     */
    void setParam1(double value);
    double getParam1() const;

    /**
     * specify when to evaluate expressions
     * Unit: eval
     * Possible Values: init (0), frame (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEval(int value);
    int getEval() const;

    Scale(const std::string& width = "", const std::string& height = "", const std::string& flags = "", bool interl = false, const std::string& size = "", int in_color_matrix = -1, int out_color_matrix = 2, int in_range = 0, int out_range = 0, int in_v_chr_pos = -513, int in_h_chr_pos = -513, int out_v_chr_pos = -513, int out_h_chr_pos = -513, int force_original_aspect_ratio = 0, int force_divisible_by = 1, double param0 = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00, double param1 = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00, int eval = 0);
    virtual ~Scale();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string width_;
    std::string height_;
    std::string flags_;
    bool interl_;
    std::string size_;
    int in_color_matrix_;
    int out_color_matrix_;
    int in_range_;
    int out_range_;
    int in_v_chr_pos_;
    int in_h_chr_pos_;
    int out_v_chr_pos_;
    int out_h_chr_pos_;
    int force_original_aspect_ratio_;
    int force_divisible_by_;
    double param0_;
    double param1_;
    int eval_;
};
