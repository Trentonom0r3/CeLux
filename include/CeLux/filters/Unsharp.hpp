#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Unsharp : public FilterBase {
public:
    /**
     * Sharpen or blur the input video.
     */
    /**
     * set luma matrix horizontal size
     * Aliases: lx
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setLuma_msize_x(int value);
    int getLuma_msize_x() const;

    /**
     * set luma matrix vertical size
     * Aliases: ly
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setLuma_msize_y(int value);
    int getLuma_msize_y() const;

    /**
     * set luma effect strength
     * Aliases: la
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setLuma_amount(float value);
    float getLuma_amount() const;

    /**
     * set chroma matrix horizontal size
     * Aliases: cx
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setChroma_msize_x(int value);
    int getChroma_msize_x() const;

    /**
     * set chroma matrix vertical size
     * Aliases: cy
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setChroma_msize_y(int value);
    int getChroma_msize_y() const;

    /**
     * set chroma effect strength
     * Aliases: ca
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setChroma_amount(float value);
    float getChroma_amount() const;

    /**
     * set alpha matrix horizontal size
     * Aliases: ax
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setAlpha_msize_x(int value);
    int getAlpha_msize_x() const;

    /**
     * set alpha matrix vertical size
     * Aliases: ay
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setAlpha_msize_y(int value);
    int getAlpha_msize_y() const;

    /**
     * set alpha effect strength
     * Aliases: aa
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setAlpha_amount(float value);
    float getAlpha_amount() const;

    Unsharp(int luma_msize_x = 5, int luma_msize_y = 5, float luma_amount = 1.00, int chroma_msize_x = 5, int chroma_msize_y = 5, float chroma_amount = 0.00, int alpha_msize_x = 5, int alpha_msize_y = 5, float alpha_amount = 0.00);
    virtual ~Unsharp();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int luma_msize_x_;
    int luma_msize_y_;
    float luma_amount_;
    int chroma_msize_x_;
    int chroma_msize_y_;
    float chroma_amount_;
    int alpha_msize_x_;
    int alpha_msize_y_;
    float alpha_amount_;
};
