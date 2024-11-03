#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Bm3d : public FilterBase {
public:
    /**
     * Block-Matching 3D denoiser.
     */
    /**
     * set denoising strength
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setSigma(float value);
    float getSigma() const;

    /**
     * set size of local patch
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setBlock(int value);
    int getBlock() const;

    /**
     * set sliding step for processing blocks
     * Type: Integer
     * Required: No
     * Default: 4
     */
    void setBstep(int value);
    int getBstep() const;

    /**
     * set maximal number of similar blocks
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setGroup(int value);
    int getGroup() const;

    /**
     * set block matching range
     * Type: Integer
     * Required: No
     * Default: 9
     */
    void setRange(int value);
    int getRange() const;

    /**
     * set step for block matching
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMstep(int value);
    int getMstep() const;

    /**
     * set threshold of mean square error for block matching
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setThmse(float value);
    float getThmse() const;

    /**
     * set hard threshold for 3D transfer domain
     * Type: Float
     * Required: No
     * Default: 2.70
     */
    void setHdthr(float value);
    float getHdthr() const;

    /**
     * set filtering estimation mode
     * Unit: mode
     * Possible Values: basic (0), final (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEstim(int value);
    int getEstim() const;

    /**
     * have reference stream
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setRef(bool value);
    bool getRef() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    Bm3d(float sigma = 1.00, int block = 16, int bstep = 4, int group = 1, int range = 9, int mstep = 1, float thmse = 0.00, float hdthr = 2.70, int estim = 0, bool ref = false, int planes = 7);
    virtual ~Bm3d();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float sigma_;
    int block_;
    int bstep_;
    int group_;
    int range_;
    int mstep_;
    float thmse_;
    float hdthr_;
    int estim_;
    bool ref_;
    int planes_;
};
