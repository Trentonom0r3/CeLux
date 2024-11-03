#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Selectivecolor : public FilterBase {
public:
    /**
     * Apply CMYK adjustments to specific color ranges.
     */
    /**
     * select correction method
     * Unit: correction_method
     * Possible Values: absolute (0), relative (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCorrection_method(int value);
    int getCorrection_method() const;

    /**
     * adjust red regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setReds(const std::string& value);
    std::string getReds() const;

    /**
     * adjust yellow regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setYellows(const std::string& value);
    std::string getYellows() const;

    /**
     * adjust green regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setGreens(const std::string& value);
    std::string getGreens() const;

    /**
     * adjust cyan regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setCyans(const std::string& value);
    std::string getCyans() const;

    /**
     * adjust blue regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setBlues(const std::string& value);
    std::string getBlues() const;

    /**
     * adjust magenta regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setMagentas(const std::string& value);
    std::string getMagentas() const;

    /**
     * adjust white regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setWhites(const std::string& value);
    std::string getWhites() const;

    /**
     * adjust neutral regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setNeutrals(const std::string& value);
    std::string getNeutrals() const;

    /**
     * adjust black regions
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setBlacks(const std::string& value);
    std::string getBlacks() const;

    /**
     * set Photoshop selectivecolor file name
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setPsfile(const std::string& value);
    std::string getPsfile() const;

    Selectivecolor(int correction_method = 0, const std::string& reds = "", const std::string& yellows = "", const std::string& greens = "", const std::string& cyans = "", const std::string& blues = "", const std::string& magentas = "", const std::string& whites = "", const std::string& neutrals = "", const std::string& blacks = "", const std::string& psfile = "");
    virtual ~Selectivecolor();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int correction_method_;
    std::string reds_;
    std::string yellows_;
    std::string greens_;
    std::string cyans_;
    std::string blues_;
    std::string magentas_;
    std::string whites_;
    std::string neutrals_;
    std::string blacks_;
    std::string psfile_;
};
