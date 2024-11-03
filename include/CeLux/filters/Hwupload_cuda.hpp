#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hwupload_cuda : public FilterBase {
public:
    /**
     * Upload a system memory frame to a CUDA device.
     */
    /**
     * Number of the device to use
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDevice(int value);
    int getDevice() const;

    Hwupload_cuda(int device = 0);
    virtual ~Hwupload_cuda();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int device_;
};
