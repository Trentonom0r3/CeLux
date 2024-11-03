#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Yadif : public FilterBase {
public:
    /**
     * Deinterlace the input image.
     */
    /**
     * specify the interlacing mode
     * Unit: mode
     * Possible Values: send_frame (0), send_field (1), send_frame_nospatial (2), send_field_nospatial (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * specify the assumed picture field parity
     * Unit: parity
     * Possible Values: tff (0), bff (1), auto (-1)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setParity(int value);
    int getParity() const;

    /**
     * specify which frames to deinterlace
     * Unit: deint
     * Possible Values: all (0), interlaced (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDeint(int value);
    int getDeint() const;

    Yadif(int mode = 0, int parity = -1, int deint = 0);
    virtual ~Yadif();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    int parity_;
    int deint_;
};
