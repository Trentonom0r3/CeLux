#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ssim360 : public FilterBase {
public:
    /**
     * Calculate the SSIM between two 360 video streams.
     */
    /**
     * Set file where to store per-frame difference information
     * Aliases: f
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setStats_file(const std::string& value);
    std::string getStats_file() const;

    /**
     * Specifies if non-luma channels must be computed
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setCompute_chroma(int value);
    int getCompute_chroma() const;

    /**
     * Specifies the number of frames to be skipped from evaluation, for every evaluated frame
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFrame_skip_ratio(int value);
    int getFrame_skip_ratio() const;

    /**
     * projection of the reference video
     * Unit: projection
     * Possible Values: e (4), equirect (4), c3x2 (0), c2x3 (1), barrel (2), barrelsplit (3)
     * Type: Integer
     * Required: No
     * Default: 4
     */
    void setRef_projection(int value);
    int getRef_projection() const;

    /**
     * projection of the main video
     * Unit: projection
     * Possible Values: e (4), equirect (4), c3x2 (0), c2x3 (1), barrel (2), barrelsplit (3)
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setMain_projection(int value);
    int getMain_projection() const;

    /**
     * stereo format of the reference video
     * Unit: stereo_format
     * Possible Values: mono (2), tb (0), lr (1)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setRef_stereo(int value);
    int getRef_stereo() const;

    /**
     * stereo format of main video
     * Unit: stereo_format
     * Possible Values: mono (2), tb (0), lr (1)
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setMain_stereo(int value);
    int getMain_stereo() const;

    /**
     * Expansion (padding) coefficient for each cube face of the reference video
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRef_pad(float value);
    float getRef_pad() const;

    /**
     * Expansion (padding) coeffiecient for each cube face of the main video
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setMain_pad(float value);
    float getMain_pad() const;

    /**
     * Specifies if the tape based SSIM 360 algorithm must be used independent of the input video types
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setUse_tape(int value);
    int getUse_tape() const;

    /**
     * Heatmap data for view-based evaluation. For heatmap file format, please refer to EntSphericalVideoHeatmapData.
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setHeatmap_str(const std::string& value);
    std::string getHeatmap_str() const;

    /**
     * Default heatmap dimension. Will be used when dimension is not specified in heatmap data.
     * Type: Integer
     * Required: No
     * Default: 32
     */
    void setDefault_heatmap_width(int value);
    int getDefault_heatmap_width() const;

    /**
     * Default heatmap dimension. Will be used when dimension is not specified in heatmap data.
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setDefault_heatmap_height(int value);
    int getDefault_heatmap_height() const;

    Ssim360(const std::string& stats_file = "", int compute_chroma = 1, int frame_skip_ratio = 0, int ref_projection = 4, int main_projection = 5, int ref_stereo = 2, int main_stereo = 3, float ref_pad = 0.00, float main_pad = 0.00, int use_tape = 0, const std::string& heatmap_str = "", int default_heatmap_width = 32, int default_heatmap_height = 16);
    virtual ~Ssim360();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string stats_file_;
    int compute_chroma_;
    int frame_skip_ratio_;
    int ref_projection_;
    int main_projection_;
    int ref_stereo_;
    int main_stereo_;
    float ref_pad_;
    float main_pad_;
    int use_tape_;
    std::string heatmap_str_;
    int default_heatmap_width_;
    int default_heatmap_height_;
};
