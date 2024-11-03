#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class A3dscope : public FilterBase {
public:
    /**
     * Convert input audio to 3d scope video output.
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
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: hd720
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set camera FoV
     * Type: Float
     * Required: No
     * Default: 90.00
     */
    void setFov(float value);
    float getFov() const;

    /**
     * set camera roll
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRoll(float value);
    float getRoll() const;

    /**
     * set camera pitch
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setPitch(float value);
    float getPitch() const;

    /**
     * set camera yaw
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setYaw(float value);
    float getYaw() const;

    /**
     * set camera zoom
     * Aliases: xzoom, yzoom
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setZzoom(float value);
    float getZzoom() const;

    /**
     * set camera position
     * Aliases: xpos, ypos
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setZpos(float value);
    float getZpos() const;

    /**
     * set length
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setLength(int value);
    int getLength() const;

    A3dscope(std::pair<int, int> rate = std::make_pair<int, int>(0, 1), std::pair<int, int> size = std::make_pair<int, int>(0, 1), float fov = 90.00, float roll = 0.00, float pitch = 0.00, float yaw = 0.00, float zzoom = 1.00, float zpos = 0.00, int length = 15);
    virtual ~A3dscope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> rate_;
    std::pair<int, int> size_;
    float fov_;
    float roll_;
    float pitch_;
    float yaw_;
    float zzoom_;
    float zpos_;
    int length_;
};
