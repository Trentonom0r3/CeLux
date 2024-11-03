#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class V360 : public FilterBase {
public:
    /**
     * Convert 360 projection of video.
     */
    /**
     * set input projection
     * Unit: in
     * Possible Values: e (0), equirect (0), c3x2 (1), c6x1 (2), eac (3), dfisheye (5), flat (4), rectilinear (4), gnomonic (4), barrel (6), fb (6), c1x6 (7), sg (8), mercator (9), ball (10), hammer (11), sinusoidal (12), fisheye (13), pannini (14), cylindrical (15), tetrahedron (17), barrelsplit (18), tsp (19), hequirect (20), he (20), equisolid (21), og (22), octahedron (23), cylindricalea (24)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setInput(int value);
    int getInput() const;

    /**
     * set output projection
     * Unit: out
     * Possible Values: e (0), equirect (0), c3x2 (1), c6x1 (2), eac (3), dfisheye (5), flat (4), rectilinear (4), gnomonic (4), barrel (6), fb (6), c1x6 (7), sg (8), mercator (9), ball (10), hammer (11), sinusoidal (12), fisheye (13), pannini (14), cylindrical (15), perspective (16), tetrahedron (17), barrelsplit (18), tsp (19), hequirect (20), he (20), equisolid (21), og (22), octahedron (23), cylindricalea (24)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setOutput(int value);
    int getOutput() const;

    /**
     * set interpolation method
     * Unit: interp
     * Possible Values: near (0), nearest (0), line (1), linear (1), lagrange9 (2), cube (3), cubic (3), lanc (4), lanczos (4), sp16 (5), spline16 (5), gauss (6), gaussian (6), mitchell (7)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setInterp(int value);
    int getInterp() const;

    /**
     * output width
     * Unit: w
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOutputWidth(int value);
    int getOutputWidth() const;

    /**
     * output height
     * Unit: h
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOutputHeight(int value);
    int getOutputHeight() const;

    /**
     * input stereo format
     * Unit: stereo
     * Possible Values: 2d (0), sbs (1), tb (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setIn_stereo(int value);
    int getIn_stereo() const;

    /**
     * output stereo format
     * Unit: stereo
     * Possible Values: 2d (0), sbs (1), tb (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOut_stereo(int value);
    int getOut_stereo() const;

    /**
     * input cubemap face order
     * Unit: in_forder
     * Type: String
     * Required: No
     * Default: rludfb
     */
    void setIn_forder(const std::string& value);
    std::string getIn_forder() const;

    /**
     * output cubemap face order
     * Unit: out_forder
     * Type: String
     * Required: No
     * Default: rludfb
     */
    void setOut_forder(const std::string& value);
    std::string getOut_forder() const;

    /**
     * input cubemap face rotation
     * Unit: in_frot
     * Type: String
     * Required: No
     * Default: 000000
     */
    void setIn_frot(const std::string& value);
    std::string getIn_frot() const;

    /**
     * output cubemap face rotation
     * Unit: out_frot
     * Type: String
     * Required: No
     * Default: 000000
     */
    void setOut_frot(const std::string& value);
    std::string getOut_frot() const;

    /**
     * percent input cubemap pads
     * Unit: in_pad
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIn_pad(float value);
    float getIn_pad() const;

    /**
     * percent output cubemap pads
     * Unit: out_pad
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setOut_pad(float value);
    float getOut_pad() const;

    /**
     * fixed input cubemap pads
     * Unit: fin_pad
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFin_pad(int value);
    int getFin_pad() const;

    /**
     * fixed output cubemap pads
     * Unit: fout_pad
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFout_pad(int value);
    int getFout_pad() const;

    /**
     * yaw rotation
     * Unit: yaw
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setYaw(float value);
    float getYaw() const;

    /**
     * pitch rotation
     * Unit: pitch
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setPitch(float value);
    float getPitch() const;

    /**
     * roll rotation
     * Unit: roll
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRoll(float value);
    float getRoll() const;

    /**
     * rotation order
     * Unit: rorder
     * Type: String
     * Required: No
     * Default: ypr
     */
    void setRorder(const std::string& value);
    std::string getRorder() const;

    /**
     * output horizontal field of view
     * Unit: h_fov
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setH_fov(float value);
    float getH_fov() const;

    /**
     * output vertical field of view
     * Unit: v_fov
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setV_fov(float value);
    float getV_fov() const;

    /**
     * output diagonal field of view
     * Unit: d_fov
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setD_fov(float value);
    float getD_fov() const;

    /**
     * flip out video horizontally
     * Unit: h_flip
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setH_flip(bool value);
    bool getH_flip() const;

    /**
     * flip out video vertically
     * Unit: v_flip
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setV_flip(bool value);
    bool getV_flip() const;

    /**
     * flip out video indepth
     * Unit: d_flip
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setD_flip(bool value);
    bool getD_flip() const;

    /**
     * flip in video horizontally
     * Unit: ih_flip
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setIh_flip(bool value);
    bool getIh_flip() const;

    /**
     * flip in video vertically
     * Unit: iv_flip
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setIv_flip(bool value);
    bool getIv_flip() const;

    /**
     * transpose video input
     * Unit: in_transpose
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setIn_trans(bool value);
    bool getIn_trans() const;

    /**
     * transpose video output
     * Unit: out_transpose
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setOut_trans(bool value);
    bool getOut_trans() const;

    /**
     * input horizontal field of view
     * Unit: ih_fov
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIh_fov(float value);
    float getIh_fov() const;

    /**
     * input vertical field of view
     * Unit: iv_fov
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIv_fov(float value);
    float getIv_fov() const;

    /**
     * input diagonal field of view
     * Unit: id_fov
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setId_fov(float value);
    float getId_fov() const;

    /**
     * output horizontal off-axis offset
     * Unit: h_offset
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setH_offset(float value);
    float getH_offset() const;

    /**
     * output vertical off-axis offset
     * Unit: v_offset
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setV_offset(float value);
    float getV_offset() const;

    /**
     * build mask in alpha plane
     * Unit: alpha
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAlpha_mask(bool value);
    bool getAlpha_mask() const;

    /**
     * reset rotation
     * Unit: reset_rot
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setReset_rot(bool value);
    bool getReset_rot() const;

    V360(int input = 0, int output = 1, int interp = 1, int outputWidth = 0, int outputHeight = 0, int in_stereo = 0, int out_stereo = 0, const std::string& in_forder = "rludfb", const std::string& out_forder = "rludfb", const std::string& in_frot = "000000", const std::string& out_frot = "000000", float in_pad = 0.00, float out_pad = 0.00, int fin_pad = 0, int fout_pad = 0, float yaw = 0.00, float pitch = 0.00, float roll = 0.00, const std::string& rorder = "ypr", float h_fov = 0.00, float v_fov = 0.00, float d_fov = 0.00, bool h_flip = false, bool v_flip = false, bool d_flip = false, bool ih_flip = false, bool iv_flip = false, bool in_trans = false, bool out_trans = false, float ih_fov = 0.00, float iv_fov = 0.00, float id_fov = 0.00, float h_offset = 0.00, float v_offset = 0.00, bool alpha_mask = false, bool reset_rot = false);
    virtual ~V360();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int input_;
    int output_;
    int interp_;
    int outputWidth_;
    int outputHeight_;
    int in_stereo_;
    int out_stereo_;
    std::string in_forder_;
    std::string out_forder_;
    std::string in_frot_;
    std::string out_frot_;
    float in_pad_;
    float out_pad_;
    int fin_pad_;
    int fout_pad_;
    float yaw_;
    float pitch_;
    float roll_;
    std::string rorder_;
    float h_fov_;
    float v_fov_;
    float d_fov_;
    bool h_flip_;
    bool v_flip_;
    bool d_flip_;
    bool ih_flip_;
    bool iv_flip_;
    bool in_trans_;
    bool out_trans_;
    float ih_fov_;
    float iv_fov_;
    float id_fov_;
    float h_offset_;
    float v_offset_;
    bool alpha_mask_;
    bool reset_rot_;
};
