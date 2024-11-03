#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Zoompan : public FilterBase {
public:
    /**
     * Apply Zoom & Pan effect.
     */
    /**
     * set the zoom expression
     * Aliases: z
     * Type: String
     * Required: No
     * Default: 1
     */
    void setZoom(const std::string& value);
    std::string getZoom() const;

    /**
     * set the x expression
     * Type: String
     * Required: No
     * Default: 0
     */
    void setX(const std::string& value);
    std::string getX() const;

    /**
     * set the y expression
     * Type: String
     * Required: No
     * Default: 0
     */
    void setY(const std::string& value);
    std::string getY() const;

    /**
     * set the duration expression
     * Type: String
     * Required: No
     * Default: 90
     */
    void setDuration(const std::string& value);
    std::string getDuration() const;

    /**
     * set the output image size
     * Type: Image Size
     * Required: No
     * Default: hd720
     */
    void setOutputImageSize(const std::pair<int, int>& value);
    std::pair<int, int> getOutputImageSize() const;

    /**
     * set the output framerate
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setFps(const std::pair<int, int>& value);
    std::pair<int, int> getFps() const;

    Zoompan(const std::string& zoom = "1", const std::string& x = "0", const std::string& y = "0", const std::string& duration = "90", std::pair<int, int> outputImageSize = std::make_pair<int, int>(0, 1), std::pair<int, int> fps = std::make_pair<int, int>(0, 1));
    virtual ~Zoompan();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string zoom_;
    std::string x_;
    std::string y_;
    std::string duration_;
    std::pair<int, int> outputImageSize_;
    std::pair<int, int> fps_;
};
