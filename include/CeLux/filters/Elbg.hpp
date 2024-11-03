#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Elbg : public FilterBase {
public:
    /**
     * Apply posterize effect, using the ELBG algorithm.
     */
    /**
     * set codebook length
     * Aliases: l
     * Type: Integer
     * Required: No
     * Default: 256
     */
    void setCodebook_length(int value);
    int getCodebook_length() const;

    /**
     * set max number of steps used to compute the mapping
     * Aliases: n
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setNb_steps(int value);
    int getNb_steps() const;

    /**
     * set the random seed
     * Aliases: s
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setSeed(int64_t value);
    int64_t getSeed() const;

    /**
     * set the pal8 output
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setPal8(bool value);
    bool getPal8() const;

    /**
     * use alpha channel for mapping
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setUse_alpha(bool value);
    bool getUse_alpha() const;

    Elbg(int codebook_length = 256, int nb_steps = 1, int64_t seed = 0, bool pal8 = false, bool use_alpha = false);
    virtual ~Elbg();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int codebook_length_;
    int nb_steps_;
    int64_t seed_;
    bool pal8_;
    bool use_alpha_;
};
