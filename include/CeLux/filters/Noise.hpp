#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Noise : public FilterBase {
public:
    /**
     * Add noise.
     */
    /**
     * set component #0 noise seed
     * Aliases: c0_seed
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setAll_seed(int value);
    int getAll_seed() const;

    /**
     * set component #0 strength
     * Aliases: c0s, alls, c0_strength
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAll_strength(int value);
    int getAll_strength() const;

    /**
     * set component #0 flags
     * Aliases: allf
     * Unit: all_flags
     * Possible Values: a (8), p (16), t (2), u (1)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setAll_flags(int value);
    int getAll_flags() const;

    /**
     * set component #0 flags
     * Aliases: c0f
     * Unit: c0_flags
     * Possible Values: a (8), p (16), t (2), u (1)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setC0_flags(int value);
    int getC0_flags() const;

    /**
     * set component #1 noise seed
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setC1_seed(int value);
    int getC1_seed() const;

    /**
     * set component #1 strength
     * Aliases: c1s
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC1_strength(int value);
    int getC1_strength() const;

    /**
     * set component #1 flags
     * Aliases: c1f
     * Unit: c1_flags
     * Possible Values: a (8), p (16), t (2), u (1)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setC1_flags(int value);
    int getC1_flags() const;

    /**
     * set component #2 noise seed
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setC2_seed(int value);
    int getC2_seed() const;

    /**
     * set component #2 strength
     * Aliases: c2s
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC2_strength(int value);
    int getC2_strength() const;

    /**
     * set component #2 flags
     * Aliases: c2f
     * Unit: c2_flags
     * Possible Values: a (8), p (16), t (2), u (1)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setC2_flags(int value);
    int getC2_flags() const;

    /**
     * set component #3 noise seed
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setC3_seed(int value);
    int getC3_seed() const;

    /**
     * set component #3 strength
     * Aliases: c3s
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC3_strength(int value);
    int getC3_strength() const;

    /**
     * set component #3 flags
     * Aliases: c3f
     * Unit: c3_flags
     * Possible Values: a (8), p (16), t (2), u (1)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setC3_flags(int value);
    int getC3_flags() const;

    Noise(int all_seed = -1, int all_strength = 0, int all_flags = 0, int c0_flags = 0, int c1_seed = -1, int c1_strength = 0, int c1_flags = 0, int c2_seed = -1, int c2_strength = 0, int c2_flags = 0, int c3_seed = -1, int c3_strength = 0, int c3_flags = 0);
    virtual ~Noise();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int all_seed_;
    int all_strength_;
    int all_flags_;
    int c0_flags_;
    int c1_seed_;
    int c1_strength_;
    int c1_flags_;
    int c2_seed_;
    int c2_strength_;
    int c2_flags_;
    int c3_seed_;
    int c3_strength_;
    int c3_flags_;
};
