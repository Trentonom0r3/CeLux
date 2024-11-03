#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dctdnoiz : public FilterBase {
public:
    /**
     * Denoise frames using 2D DCT.
     */
    /**
     * set noise sigma constant
     * Aliases: s
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setSigma(float value);
    float getSigma() const;

    /**
     * set number of block overlapping pixels
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setOverlap(int value);
    int getOverlap() const;

    /**
     * set coefficient factor expression
     * Aliases: e
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setExpr(const std::string& value);
    std::string getExpr() const;

    /**
     * set the block size, expressed in bits
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setBlockSizeExpressedInBits(int value);
    int getBlockSizeExpressedInBits() const;

    Dctdnoiz(float sigma = 0.00, int overlap = -1, const std::string& expr = "", int blockSizeExpressedInBits = 3);
    virtual ~Dctdnoiz();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float sigma_;
    int overlap_;
    std::string expr_;
    int blockSizeExpressedInBits_;
};
