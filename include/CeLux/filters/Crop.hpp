#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Crop : public FilterBase {
public:
    /**
     * Crop the input video.
     */
    /**
     * set the width crop area expression
     * Aliases: w
     * Type: String
     * Required: No
     * Default: iw
     */
    void setOut_w(const std::string& value);
    std::string getOut_w() const;

    /**
     * set the height crop area expression
     * Aliases: h
     * Type: String
     * Required: No
     * Default: ih
     */
    void setOut_h(const std::string& value);
    std::string getOut_h() const;

    /**
     * set the x crop area expression
     * Type: String
     * Required: No
     * Default: (in_w-out_w)/2
     */
    void setXCropArea(const std::string& value);
    std::string getXCropArea() const;

    /**
     * set the y crop area expression
     * Type: String
     * Required: No
     * Default: (in_h-out_h)/2
     */
    void setYCropArea(const std::string& value);
    std::string getYCropArea() const;

    /**
     * keep aspect ratio
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setKeep_aspect(bool value);
    bool getKeep_aspect() const;

    /**
     * do exact cropping
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setExact(bool value);
    bool getExact() const;

    Crop(const std::string& out_w = "iw", const std::string& out_h = "ih", const std::string& xCropArea = "(in_w-out_w)/2", const std::string& yCropArea = "(in_h-out_h)/2", bool keep_aspect = false, bool exact = false);
    virtual ~Crop();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string out_w_;
    std::string out_h_;
    std::string xCropArea_;
    std::string yCropArea_;
    bool keep_aspect_;
    bool exact_;
};
