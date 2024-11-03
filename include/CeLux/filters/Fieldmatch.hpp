#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fieldmatch : public FilterBase {
public:
    /**
     * Field matching for inverse telecine.
     */
    /**
     * specify the assumed field order
     * Unit: order
     * Possible Values: auto (-1), bff (0), tff (1)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setOrder(int value);
    int getOrder() const;

    /**
     * set the matching mode or strategy to use
     * Unit: mode
     * Possible Values: pc (0), pc_n (1), pc_u (2), pc_n_ub (3), pcn (4), pcn_ub (5)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMode(int value);
    int getMode() const;

    /**
     * mark main input as a pre-processed input and activate clean source input stream
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setPpsrc(bool value);
    bool getPpsrc() const;

    /**
     * set the field to match from
     * Unit: field
     * Possible Values: auto (-1), bottom (0), top (1)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setField(int value);
    int getField() const;

    /**
     * set whether or not chroma is included during the match comparisons
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setMchroma(bool value);
    bool getMchroma() const;

    /**
     * define an exclusion band which excludes the lines between y0 and y1 from the field matching decision
     * Aliases: y0
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setY1(int value);
    int getY1() const;

    /**
     * set scene change detection threshold
     * Type: Double
     * Required: No
     * Default: 12.00
     */
    void setScthresh(double value);
    double getScthresh() const;

    /**
     * set combmatching mode
     * Unit: combmatching
     * Possible Values: none (0), sc (1), full (2)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setCombmatch(int value);
    int getCombmatch() const;

    /**
     * enable comb debug
     * Unit: dbglvl
     * Possible Values: none (0), pcn (1), pcnub (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCombdbg(int value);
    int getCombdbg() const;

    /**
     * set the area combing threshold used for combed frame detection
     * Type: Integer
     * Required: No
     * Default: 9
     */
    void setCthresh(int value);
    int getCthresh() const;

    /**
     * set whether or not chroma is considered in the combed frame decision
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setChroma(bool value);
    bool getChroma() const;

    /**
     * set the x-axis size of the window used during combed frame detection
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setBlockx(int value);
    int getBlockx() const;

    /**
     * set the y-axis size of the window used during combed frame detection
     * Type: Integer
     * Required: No
     * Default: 16
     */
    void setBlocky(int value);
    int getBlocky() const;

    /**
     * set the number of combed pixels inside any of the blocky by blockx size blocks on the frame for the frame to be detected as combed
     * Type: Integer
     * Required: No
     * Default: 80
     */
    void setCombpel(int value);
    int getCombpel() const;

    Fieldmatch(int order = -1, int mode = 1, bool ppsrc = false, int field = -1, bool mchroma = true, int y1 = 0, double scthresh = 12.00, int combmatch = 1, int combdbg = 0, int cthresh = 9, bool chroma = false, int blockx = 16, int blocky = 16, int combpel = 80);
    virtual ~Fieldmatch();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int order_;
    int mode_;
    bool ppsrc_;
    int field_;
    bool mchroma_;
    int y1_;
    double scthresh_;
    int combmatch_;
    int combdbg_;
    int cthresh_;
    bool chroma_;
    int blockx_;
    int blocky_;
    int combpel_;
};
