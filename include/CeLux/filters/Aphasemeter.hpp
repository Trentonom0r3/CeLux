#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Aphasemeter : public FilterBase {
public:
    /**
     * Convert input audio to phase meter video output.
     */
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
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 800x400
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set red contrast
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setRc(int value);
    int getRc() const;

    /**
     * set green contrast
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setGc(int value);
    int getGc() const;

    /**
     * set blue contrast
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setBc(int value);
    int getBc() const;

    /**
     * set median phase color
     * Type: String
     * Required: No
     * Default: none
     */
    void setMpc(const std::string& value);
    std::string getMpc() const;

    /**
     * set video output
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setVideo(bool value);
    bool getVideo() const;

    /**
     * set mono and out-of-phase detection output
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setPhasing(bool value);
    bool getPhasing() const;

    /**
     * set phase tolerance for mono detection
     * Aliases: t
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setTolerance(float value);
    float getTolerance() const;

    /**
     * set angle threshold for out-of-phase detection
     * Aliases: a
     * Type: Float
     * Required: No
     * Default: 170.00
     */
    void setAngle(float value);
    float getAngle() const;

    /**
     * set minimum mono or out-of-phase duration in seconds
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: 2000000
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    Aphasemeter(std::pair<int, int> rate = std::make_pair<int, int>(0, 1), std::pair<int, int> size = std::make_pair<int, int>(0, 1), int rc = 2, int gc = 7, int bc = 1, const std::string& mpc = "none", bool video = true, bool phasing = false, float tolerance = 0.00, float angle = 170.00, int64_t duration = 2000000ULL);
    virtual ~Aphasemeter();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> rate_;
    std::pair<int, int> size_;
    int rc_;
    int gc_;
    int bc_;
    std::string mpc_;
    bool video_;
    bool phasing_;
    float tolerance_;
    float angle_;
    int64_t duration_;
};
