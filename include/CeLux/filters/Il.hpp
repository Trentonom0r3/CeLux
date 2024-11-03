#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Il : public FilterBase {
public:
    /**
     * Deinterleave or interleave fields.
     */
    /**
     * select luma mode
     * Aliases: l
     * Unit: luma_mode
     * Possible Values: none (0), interleave (1), i (1), deinterleave (2), d (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setLuma_mode(int value);
    int getLuma_mode() const;

    /**
     * select chroma mode
     * Aliases: c
     * Unit: chroma_mode
     * Possible Values: none (0), interleave (1), i (1), deinterleave (2), d (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setChroma_mode(int value);
    int getChroma_mode() const;

    /**
     * select alpha mode
     * Aliases: a
     * Unit: alpha_mode
     * Possible Values: none (0), interleave (1), i (1), deinterleave (2), d (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAlpha_mode(int value);
    int getAlpha_mode() const;

    /**
     * swap luma fields
     * Aliases: ls
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setLuma_swap(bool value);
    bool getLuma_swap() const;

    /**
     * swap chroma fields
     * Aliases: cs
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setChroma_swap(bool value);
    bool getChroma_swap() const;

    /**
     * swap alpha fields
     * Aliases: as
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAlpha_swap(bool value);
    bool getAlpha_swap() const;

    Il(int luma_mode = 0, int chroma_mode = 0, int alpha_mode = 0, bool luma_swap = false, bool chroma_swap = false, bool alpha_swap = false);
    virtual ~Il();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int luma_mode_;
    int chroma_mode_;
    int alpha_mode_;
    bool luma_swap_;
    bool chroma_swap_;
    bool alpha_swap_;
};
