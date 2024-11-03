#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tblend : public FilterBase {
public:
    /**
     * Blend successive frames.
     */
    /**
     * set component #0 blend mode
     * Unit: mode
     * Possible Values: addition (1), addition128 (28), grainmerge (28), and (2), average (3), burn (4), darken (5), difference (6), difference128 (7), grainextract (7), divide (8), dodge (9), exclusion (10), extremity (32), freeze (31), glow (27), hardlight (11), hardmix (25), heat (30), lighten (12), linearlight (26), multiply (13), multiply128 (29), negation (14), normal (0), or (15), overlay (16), phoenix (17), pinlight (18), reflect (19), screen (20), softlight (21), subtract (22), vividlight (23), xor (24), softdifference (33), geometric (34), harmonic (35), bleach (36), stain (37), interpolate (38), hardoverlay (39)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC0_mode(int value);
    int getC0_mode() const;

    /**
     * set component #1 blend mode
     * Unit: mode
     * Possible Values: addition (1), addition128 (28), grainmerge (28), and (2), average (3), burn (4), darken (5), difference (6), difference128 (7), grainextract (7), divide (8), dodge (9), exclusion (10), extremity (32), freeze (31), glow (27), hardlight (11), hardmix (25), heat (30), lighten (12), linearlight (26), multiply (13), multiply128 (29), negation (14), normal (0), or (15), overlay (16), phoenix (17), pinlight (18), reflect (19), screen (20), softlight (21), subtract (22), vividlight (23), xor (24), softdifference (33), geometric (34), harmonic (35), bleach (36), stain (37), interpolate (38), hardoverlay (39)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC1_mode(int value);
    int getC1_mode() const;

    /**
     * set component #2 blend mode
     * Unit: mode
     * Possible Values: addition (1), addition128 (28), grainmerge (28), and (2), average (3), burn (4), darken (5), difference (6), difference128 (7), grainextract (7), divide (8), dodge (9), exclusion (10), extremity (32), freeze (31), glow (27), hardlight (11), hardmix (25), heat (30), lighten (12), linearlight (26), multiply (13), multiply128 (29), negation (14), normal (0), or (15), overlay (16), phoenix (17), pinlight (18), reflect (19), screen (20), softlight (21), subtract (22), vividlight (23), xor (24), softdifference (33), geometric (34), harmonic (35), bleach (36), stain (37), interpolate (38), hardoverlay (39)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC2_mode(int value);
    int getC2_mode() const;

    /**
     * set component #3 blend mode
     * Unit: mode
     * Possible Values: addition (1), addition128 (28), grainmerge (28), and (2), average (3), burn (4), darken (5), difference (6), difference128 (7), grainextract (7), divide (8), dodge (9), exclusion (10), extremity (32), freeze (31), glow (27), hardlight (11), hardmix (25), heat (30), lighten (12), linearlight (26), multiply (13), multiply128 (29), negation (14), normal (0), or (15), overlay (16), phoenix (17), pinlight (18), reflect (19), screen (20), softlight (21), subtract (22), vividlight (23), xor (24), softdifference (33), geometric (34), harmonic (35), bleach (36), stain (37), interpolate (38), hardoverlay (39)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setC3_mode(int value);
    int getC3_mode() const;

    /**
     * set blend mode for all components
     * Unit: mode
     * Possible Values: addition (1), addition128 (28), grainmerge (28), and (2), average (3), burn (4), darken (5), difference (6), difference128 (7), grainextract (7), divide (8), dodge (9), exclusion (10), extremity (32), freeze (31), glow (27), hardlight (11), hardmix (25), heat (30), lighten (12), linearlight (26), multiply (13), multiply128 (29), negation (14), normal (0), or (15), overlay (16), phoenix (17), pinlight (18), reflect (19), screen (20), softlight (21), subtract (22), vividlight (23), xor (24), softdifference (33), geometric (34), harmonic (35), bleach (36), stain (37), interpolate (38), hardoverlay (39)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setAll_mode(int value);
    int getAll_mode() const;

    /**
     * set color component #0 expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setC0_expr(const std::string& value);
    std::string getC0_expr() const;

    /**
     * set color component #1 expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setC1_expr(const std::string& value);
    std::string getC1_expr() const;

    /**
     * set color component #2 expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setC2_expr(const std::string& value);
    std::string getC2_expr() const;

    /**
     * set color component #3 expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setC3_expr(const std::string& value);
    std::string getC3_expr() const;

    /**
     * set expression for all color components
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setAll_expr(const std::string& value);
    std::string getAll_expr() const;

    /**
     * set color component #0 opacity
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setC0_opacity(double value);
    double getC0_opacity() const;

    /**
     * set color component #1 opacity
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setC1_opacity(double value);
    double getC1_opacity() const;

    /**
     * set color component #2 opacity
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setC2_opacity(double value);
    double getC2_opacity() const;

    /**
     * set color component #3 opacity
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setC3_opacity(double value);
    double getC3_opacity() const;

    /**
     * set opacity for all color components
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setAll_opacity(double value);
    double getAll_opacity() const;

    Tblend(int c0_mode = 0, int c1_mode = 0, int c2_mode = 0, int c3_mode = 0, int all_mode = -1, const std::string& c0_expr = "", const std::string& c1_expr = "", const std::string& c2_expr = "", const std::string& c3_expr = "", const std::string& all_expr = "", double c0_opacity = 1.00, double c1_opacity = 1.00, double c2_opacity = 1.00, double c3_opacity = 1.00, double all_opacity = 1.00);
    virtual ~Tblend();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int c0_mode_;
    int c1_mode_;
    int c2_mode_;
    int c3_mode_;
    int all_mode_;
    std::string c0_expr_;
    std::string c1_expr_;
    std::string c2_expr_;
    std::string c3_expr_;
    std::string all_expr_;
    double c0_opacity_;
    double c1_opacity_;
    double c2_opacity_;
    double c3_opacity_;
    double all_opacity_;
};
