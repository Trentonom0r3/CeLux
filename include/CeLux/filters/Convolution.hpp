#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Convolution : public FilterBase {
public:
    /**
     * Apply convolution filter.
     */
    /**
     * set matrix for 1st plane
     * Type: String
     * Required: No
     * Default: 0 0 0 0 1 0 0 0 0
     */
    void set_0m(const std::string& value);
    std::string get_0m() const;

    /**
     * set matrix for 2nd plane
     * Type: String
     * Required: No
     * Default: 0 0 0 0 1 0 0 0 0
     */
    void set_1m(const std::string& value);
    std::string get_1m() const;

    /**
     * set matrix for 3rd plane
     * Type: String
     * Required: No
     * Default: 0 0 0 0 1 0 0 0 0
     */
    void set_2m(const std::string& value);
    std::string get_2m() const;

    /**
     * set matrix for 4th plane
     * Type: String
     * Required: No
     * Default: 0 0 0 0 1 0 0 0 0
     */
    void set_3m(const std::string& value);
    std::string get_3m() const;

    /**
     * set rdiv for 1st plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_0rdiv(float value);
    float get_0rdiv() const;

    /**
     * set rdiv for 2nd plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_1rdiv(float value);
    float get_1rdiv() const;

    /**
     * set rdiv for 3rd plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_2rdiv(float value);
    float get_2rdiv() const;

    /**
     * set rdiv for 4th plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_3rdiv(float value);
    float get_3rdiv() const;

    /**
     * set bias for 1st plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_0bias(float value);
    float get_0bias() const;

    /**
     * set bias for 2nd plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_1bias(float value);
    float get_1bias() const;

    /**
     * set bias for 3rd plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_2bias(float value);
    float get_2bias() const;

    /**
     * set bias for 4th plane
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void set_3bias(float value);
    float get_3bias() const;

    /**
     * set matrix mode for 1st plane
     * Unit: mode
     * Possible Values: square (0), row (1), column (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void set_0mode(int value);
    int get_0mode() const;

    /**
     * set matrix mode for 2nd plane
     * Unit: mode
     * Possible Values: square (0), row (1), column (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void set_1mode(int value);
    int get_1mode() const;

    /**
     * set matrix mode for 3rd plane
     * Unit: mode
     * Possible Values: square (0), row (1), column (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void set_2mode(int value);
    int get_2mode() const;

    /**
     * set matrix mode for 4th plane
     * Unit: mode
     * Possible Values: square (0), row (1), column (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void set_3mode(int value);
    int get_3mode() const;

    Convolution(const std::string& _0m = "0 0 0 0 1 0 0 0 0", const std::string& _1m = "0 0 0 0 1 0 0 0 0", const std::string& _2m = "0 0 0 0 1 0 0 0 0", const std::string& _3m = "0 0 0 0 1 0 0 0 0", float _0rdiv = 0.00, float _1rdiv = 0.00, float _2rdiv = 0.00, float _3rdiv = 0.00, float _0bias = 0.00, float _1bias = 0.00, float _2bias = 0.00, float _3bias = 0.00, int _0mode = 0, int _1mode = 0, int _2mode = 0, int _3mode = 0);
    virtual ~Convolution();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string _0m_;
    std::string _1m_;
    std::string _2m_;
    std::string _3m_;
    float _0rdiv_;
    float _1rdiv_;
    float _2rdiv_;
    float _3rdiv_;
    float _0bias_;
    float _1bias_;
    float _2bias_;
    float _3bias_;
    int _0mode_;
    int _1mode_;
    int _2mode_;
    int _3mode_;
};
