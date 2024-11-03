#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Zoneplate : public FilterBase {
public:
    /**
     * Generate zone-plate.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 320x240
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set video duration
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: -1
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    /**
     * set video sample aspect ratio
     * Type: Rational
     * Required: No
     * Default: 0
     */
    void setSar(const std::pair<int, int>& value);
    std::pair<int, int> getSar() const;

    /**
     * set LUT precision
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setPrecision(int value);
    int getPrecision() const;

    /**
     * set X-axis offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setXo(int value);
    int getXo() const;

    /**
     * set Y-axis offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setYo(int value);
    int getYo() const;

    /**
     * set T-axis offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setTo(int value);
    int getTo() const;

    /**
     * set 0-order phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setK0(int value);
    int getK0() const;

    /**
     * set 1-order X-axis phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKx(int value);
    int getKx() const;

    /**
     * set 1-order Y-axis phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKy(int value);
    int getKy() const;

    /**
     * set 1-order T-axis phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKt(int value);
    int getKt() const;

    /**
     * set X-axis*T-axis product phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKxt(int value);
    int getKxt() const;

    /**
     * set Y-axis*T-axis product phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKyt(int value);
    int getKyt() const;

    /**
     * set X-axis*Y-axis product phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKxy(int value);
    int getKxy() const;

    /**
     * set 2-order X-axis phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKx2(int value);
    int getKx2() const;

    /**
     * set 2-order Y-axis phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKy2(int value);
    int getKy2() const;

    /**
     * set 2-order T-axis phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKt2(int value);
    int getKt2() const;

    /**
     * set 0-order U-color phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKu(int value);
    int getKu() const;

    /**
     * set 0-order V-color phase
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setKv(int value);
    int getKv() const;

    Zoneplate(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int64_t duration = 0, std::pair<int, int> sar = std::make_pair<int, int>(0, 1), int precision = 10, int xo = 0, int yo = 0, int to = 0, int k0 = 0, int kx = 0, int ky = 0, int kt = 0, int kxt = 0, int kyt = 0, int kxy = 0, int kx2 = 0, int ky2 = 0, int kt2 = 0, int ku = 0, int kv = 0);
    virtual ~Zoneplate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    int64_t duration_;
    std::pair<int, int> sar_;
    int precision_;
    int xo_;
    int yo_;
    int to_;
    int k0_;
    int kx_;
    int ky_;
    int kt_;
    int kxt_;
    int kyt_;
    int kxy_;
    int kx2_;
    int ky2_;
    int kt2_;
    int ku_;
    int kv_;
};
