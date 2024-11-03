#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hwupload : public FilterBase {
public:
    /**
     * Upload a normal frame to a hardware frame
     */
    /**
     * Derive a new device of this type
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setDerive_device(const std::string& value);
    std::string getDerive_device() const;

    Hwupload(const std::string& derive_device = "");
    virtual ~Hwupload();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string derive_device_;
};
