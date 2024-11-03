#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colormap : public FilterBase {
public:
    /**
     * Apply custom Color Maps to video stream.
     */
    /**
     * set patch size
     * Type: Image Size
     * Required: No
     * Default: 64x64
     */
    void setPatch_size(const std::pair<int, int>& value);
    std::pair<int, int> getPatch_size() const;

    /**
     * set number of patches
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setNb_patches(int value);
    int getNb_patches() const;

    /**
     * set the target type used
     * Unit: type
     * Possible Values: relative (0), absolute (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setType(int value);
    int getType() const;

    /**
     * set the kernel used for measuring color difference
     * Unit: kernel
     * Possible Values: euclidean (0), weuclidean (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKernel(int value);
    int getKernel() const;

    Colormap(std::pair<int, int> patch_size = std::make_pair<int, int>(0, 1), int nb_patches = 0, int type = 1, int kernel = 0);
    virtual ~Colormap();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> patch_size_;
    int nb_patches_;
    int type_;
    int kernel_;
};
