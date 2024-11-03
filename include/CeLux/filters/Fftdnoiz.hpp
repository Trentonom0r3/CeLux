#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fftdnoiz : public FilterBase {
public:
    /**
     * Denoise frames using 3D FFT.
     */
    /**
     * set denoise strength
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setSigma(float value);
    float getSigma() const;

    /**
     * set amount of denoising
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setAmount(float value);
    float getAmount() const;

    /**
     * set block size
     * Type: Integer
     * Required: No
     * Default: 32
     */
    void setBlock(int value);
    int getBlock() const;

    /**
     * set block overlap
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setOverlap(float value);
    float getOverlap() const;

    /**
     * set method of denoising
     * Unit: method
     * Possible Values: wiener (0), hard (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMethod(int value);
    int getMethod() const;

    /**
     * set number of previous frames for temporal denoising
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPrev(int value);
    int getPrev() const;

    /**
     * set number of next frames for temporal denoising
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setNext(int value);
    int getNext() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set window function
     * Unit: win_func
     * Possible Values: rect (0), bartlett (4), hann (1), hanning (1), hamming (2), blackman (3), welch (5), flattop (6), bharris (7), bnuttall (8), bhann (11), sine (9), nuttall (10), lanczos (12), gauss (13), tukey (14), dolph (15), cauchy (16), parzen (17), poisson (18), bohman (19), kaiser (20)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setWindow(int value);
    int getWindow() const;

    Fftdnoiz(float sigma = 1.00, float amount = 1.00, int block = 32, float overlap = 0.50, int method = 0, int prev = 0, int next = 0, int planes = 7, int window = 1);
    virtual ~Fftdnoiz();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float sigma_;
    float amount_;
    int block_;
    float overlap_;
    int method_;
    int prev_;
    int next_;
    int planes_;
    int window_;
};
