#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Minterpolate : public FilterBase {
public:
    /**
     * Frame rate conversion using Motion Interpolation.
     */
    /**
     * output's frame rate
     * Type: Video Rate
     * Required: No
     * Default: 40544.8
     */
    void setFps(const std::pair<int, int>& value);
    std::pair<int, int> getFps() const;

    /**
     * motion interpolation mode
     * Unit: mi_mode
     * Possible Values: dup (0), blend (1), mci (2)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setMi_mode(int value);
    int getMi_mode() const;

    /**
     * motion compensation mode
     * Unit: mc_mode
     * Possible Values: obmc (0), aobmc (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMc_mode(int value);
    int getMc_mode() const;

    /**
     * motion estimation mode
     * Unit: me_mode
     * Possible Values: bidir (0), bilat (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMe_mode(int value);
    int getMe_mode() const;

    /**
     * motion estimation method
     * Unit: me
     * Possible Values: esa (1), tss (2), tdls (3), ntss (4), fss (5), ds (6), hexbs (7), epzs (8), umh (9)
     * Type: Integer
     * Required: No
     * Default: 8
     */
    void setMe(int value);
    int getMe() const;

    /**
     * macroblock size
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setMb_size(int value);
    int getMb_size() const;

    /**
     * search parameter
     * Type: Integer
     * Required: No
     * Default: 32
     */
    void setSearch_param(int value);
    int getSearch_param() const;

    /**
     * variable-size block motion compensation
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setVsbmc(int value);
    int getVsbmc() const;

    /**
     * scene change detection method
     * Unit: scene
     * Possible Values: none (0), fdiff (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setScd(int value);
    int getScd() const;

    /**
     * scene change threshold
     * Type: Double
     * Required: No
     * Default: 10.00
     */
    void setScd_threshold(double value);
    double getScd_threshold() const;

    Minterpolate(std::pair<int, int> fps = std::make_pair<int, int>(0, 1), int mi_mode = 2, int mc_mode = 0, int me_mode = 1, int me = 8, int mb_size = 16, int search_param = 32, int vsbmc = 0, int scd = 1, double scd_threshold = 10.00);
    virtual ~Minterpolate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> fps_;
    int mi_mode_;
    int mc_mode_;
    int me_mode_;
    int me_;
    int mb_size_;
    int search_param_;
    int vsbmc_;
    int scd_;
    double scd_threshold_;
};
