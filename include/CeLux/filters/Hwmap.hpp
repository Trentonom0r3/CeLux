#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hwmap : public FilterBase {
public:
    /**
     * Map hardware frames
     */
    /**
     * Frame mapping mode
     * Unit: mode
     * Possible Values: read (1), write (2), overwrite (4), direct (8)
     * Type: Flags
     * Required: No
     * Default: 3
     */
    void setMode(int value);
    int getMode() const;

    /**
     * Derive a new device of this type
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setDerive_device(const std::string& value);
    std::string getDerive_device() const;

    /**
     * Map in reverse (create and allocate in the sink)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setReverse(int value);
    int getReverse() const;

    Hwmap(int mode = 3, const std::string& derive_device = "", int reverse = 0);
    virtual ~Hwmap();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    std::string derive_device_;
    int reverse_;
};
