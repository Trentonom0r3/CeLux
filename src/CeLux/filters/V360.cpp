#include "V360.hpp"
#include <sstream>

V360::V360(int input, int output, int interp, int outputWidth, int outputHeight, int in_stereo, int out_stereo, const std::string& in_forder, const std::string& out_forder, const std::string& in_frot, const std::string& out_frot, float in_pad, float out_pad, int fin_pad, int fout_pad, float yaw, float pitch, float roll, const std::string& rorder, float h_fov, float v_fov, float d_fov, bool h_flip, bool v_flip, bool d_flip, bool ih_flip, bool iv_flip, bool in_trans, bool out_trans, float ih_fov, float iv_fov, float id_fov, float h_offset, float v_offset, bool alpha_mask, bool reset_rot) {
    // Initialize member variables from parameters
    this->input_ = input;
    this->output_ = output;
    this->interp_ = interp;
    this->outputWidth_ = outputWidth;
    this->outputHeight_ = outputHeight;
    this->in_stereo_ = in_stereo;
    this->out_stereo_ = out_stereo;
    this->in_forder_ = in_forder;
    this->out_forder_ = out_forder;
    this->in_frot_ = in_frot;
    this->out_frot_ = out_frot;
    this->in_pad_ = in_pad;
    this->out_pad_ = out_pad;
    this->fin_pad_ = fin_pad;
    this->fout_pad_ = fout_pad;
    this->yaw_ = yaw;
    this->pitch_ = pitch;
    this->roll_ = roll;
    this->rorder_ = rorder;
    this->h_fov_ = h_fov;
    this->v_fov_ = v_fov;
    this->d_fov_ = d_fov;
    this->h_flip_ = h_flip;
    this->v_flip_ = v_flip;
    this->d_flip_ = d_flip;
    this->ih_flip_ = ih_flip;
    this->iv_flip_ = iv_flip;
    this->in_trans_ = in_trans;
    this->out_trans_ = out_trans;
    this->ih_fov_ = ih_fov;
    this->iv_fov_ = iv_fov;
    this->id_fov_ = id_fov;
    this->h_offset_ = h_offset;
    this->v_offset_ = v_offset;
    this->alpha_mask_ = alpha_mask;
    this->reset_rot_ = reset_rot;
}

V360::~V360() {
    // Destructor implementation (if needed)
}

void V360::setInput(int value) {
    input_ = value;
}

int V360::getInput() const {
    return input_;
}

void V360::setOutput(int value) {
    output_ = value;
}

int V360::getOutput() const {
    return output_;
}

void V360::setInterp(int value) {
    interp_ = value;
}

int V360::getInterp() const {
    return interp_;
}

void V360::setOutputWidth(int value) {
    outputWidth_ = value;
}

int V360::getOutputWidth() const {
    return outputWidth_;
}

void V360::setOutputHeight(int value) {
    outputHeight_ = value;
}

int V360::getOutputHeight() const {
    return outputHeight_;
}

void V360::setIn_stereo(int value) {
    in_stereo_ = value;
}

int V360::getIn_stereo() const {
    return in_stereo_;
}

void V360::setOut_stereo(int value) {
    out_stereo_ = value;
}

int V360::getOut_stereo() const {
    return out_stereo_;
}

void V360::setIn_forder(const std::string& value) {
    in_forder_ = value;
}

std::string V360::getIn_forder() const {
    return in_forder_;
}

void V360::setOut_forder(const std::string& value) {
    out_forder_ = value;
}

std::string V360::getOut_forder() const {
    return out_forder_;
}

void V360::setIn_frot(const std::string& value) {
    in_frot_ = value;
}

std::string V360::getIn_frot() const {
    return in_frot_;
}

void V360::setOut_frot(const std::string& value) {
    out_frot_ = value;
}

std::string V360::getOut_frot() const {
    return out_frot_;
}

void V360::setIn_pad(float value) {
    in_pad_ = value;
}

float V360::getIn_pad() const {
    return in_pad_;
}

void V360::setOut_pad(float value) {
    out_pad_ = value;
}

float V360::getOut_pad() const {
    return out_pad_;
}

void V360::setFin_pad(int value) {
    fin_pad_ = value;
}

int V360::getFin_pad() const {
    return fin_pad_;
}

void V360::setFout_pad(int value) {
    fout_pad_ = value;
}

int V360::getFout_pad() const {
    return fout_pad_;
}

void V360::setYaw(float value) {
    yaw_ = value;
}

float V360::getYaw() const {
    return yaw_;
}

void V360::setPitch(float value) {
    pitch_ = value;
}

float V360::getPitch() const {
    return pitch_;
}

void V360::setRoll(float value) {
    roll_ = value;
}

float V360::getRoll() const {
    return roll_;
}

void V360::setRorder(const std::string& value) {
    rorder_ = value;
}

std::string V360::getRorder() const {
    return rorder_;
}

void V360::setH_fov(float value) {
    h_fov_ = value;
}

float V360::getH_fov() const {
    return h_fov_;
}

void V360::setV_fov(float value) {
    v_fov_ = value;
}

float V360::getV_fov() const {
    return v_fov_;
}

void V360::setD_fov(float value) {
    d_fov_ = value;
}

float V360::getD_fov() const {
    return d_fov_;
}

void V360::setH_flip(bool value) {
    h_flip_ = value;
}

bool V360::getH_flip() const {
    return h_flip_;
}

void V360::setV_flip(bool value) {
    v_flip_ = value;
}

bool V360::getV_flip() const {
    return v_flip_;
}

void V360::setD_flip(bool value) {
    d_flip_ = value;
}

bool V360::getD_flip() const {
    return d_flip_;
}

void V360::setIh_flip(bool value) {
    ih_flip_ = value;
}

bool V360::getIh_flip() const {
    return ih_flip_;
}

void V360::setIv_flip(bool value) {
    iv_flip_ = value;
}

bool V360::getIv_flip() const {
    return iv_flip_;
}

void V360::setIn_trans(bool value) {
    in_trans_ = value;
}

bool V360::getIn_trans() const {
    return in_trans_;
}

void V360::setOut_trans(bool value) {
    out_trans_ = value;
}

bool V360::getOut_trans() const {
    return out_trans_;
}

void V360::setIh_fov(float value) {
    ih_fov_ = value;
}

float V360::getIh_fov() const {
    return ih_fov_;
}

void V360::setIv_fov(float value) {
    iv_fov_ = value;
}

float V360::getIv_fov() const {
    return iv_fov_;
}

void V360::setId_fov(float value) {
    id_fov_ = value;
}

float V360::getId_fov() const {
    return id_fov_;
}

void V360::setH_offset(float value) {
    h_offset_ = value;
}

float V360::getH_offset() const {
    return h_offset_;
}

void V360::setV_offset(float value) {
    v_offset_ = value;
}

float V360::getV_offset() const {
    return v_offset_;
}

void V360::setAlpha_mask(bool value) {
    alpha_mask_ = value;
}

bool V360::getAlpha_mask() const {
    return alpha_mask_;
}

void V360::setReset_rot(bool value) {
    reset_rot_ = value;
}

bool V360::getReset_rot() const {
    return reset_rot_;
}

std::string V360::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "v360";

    bool first = true;

    if (input_ != 0) {
        desc << (first ? "=" : ":") << "input=" << input_;
        first = false;
    }
    if (output_ != 1) {
        desc << (first ? "=" : ":") << "output=" << output_;
        first = false;
    }
    if (interp_ != 1) {
        desc << (first ? "=" : ":") << "interp=" << interp_;
        first = false;
    }
    if (outputWidth_ != 0) {
        desc << (first ? "=" : ":") << "w=" << outputWidth_;
        first = false;
    }
    if (outputHeight_ != 0) {
        desc << (first ? "=" : ":") << "h=" << outputHeight_;
        first = false;
    }
    if (in_stereo_ != 0) {
        desc << (first ? "=" : ":") << "in_stereo=" << in_stereo_;
        first = false;
    }
    if (out_stereo_ != 0) {
        desc << (first ? "=" : ":") << "out_stereo=" << out_stereo_;
        first = false;
    }
    if (in_forder_ != "rludfb") {
        desc << (first ? "=" : ":") << "in_forder=" << in_forder_;
        first = false;
    }
    if (out_forder_ != "rludfb") {
        desc << (first ? "=" : ":") << "out_forder=" << out_forder_;
        first = false;
    }
    if (in_frot_ != "000000") {
        desc << (first ? "=" : ":") << "in_frot=" << in_frot_;
        first = false;
    }
    if (out_frot_ != "000000") {
        desc << (first ? "=" : ":") << "out_frot=" << out_frot_;
        first = false;
    }
    if (in_pad_ != 0.00) {
        desc << (first ? "=" : ":") << "in_pad=" << in_pad_;
        first = false;
    }
    if (out_pad_ != 0.00) {
        desc << (first ? "=" : ":") << "out_pad=" << out_pad_;
        first = false;
    }
    if (fin_pad_ != 0) {
        desc << (first ? "=" : ":") << "fin_pad=" << fin_pad_;
        first = false;
    }
    if (fout_pad_ != 0) {
        desc << (first ? "=" : ":") << "fout_pad=" << fout_pad_;
        first = false;
    }
    if (yaw_ != 0.00) {
        desc << (first ? "=" : ":") << "yaw=" << yaw_;
        first = false;
    }
    if (pitch_ != 0.00) {
        desc << (first ? "=" : ":") << "pitch=" << pitch_;
        first = false;
    }
    if (roll_ != 0.00) {
        desc << (first ? "=" : ":") << "roll=" << roll_;
        first = false;
    }
    if (rorder_ != "ypr") {
        desc << (first ? "=" : ":") << "rorder=" << rorder_;
        first = false;
    }
    if (h_fov_ != 0.00) {
        desc << (first ? "=" : ":") << "h_fov=" << h_fov_;
        first = false;
    }
    if (v_fov_ != 0.00) {
        desc << (first ? "=" : ":") << "v_fov=" << v_fov_;
        first = false;
    }
    if (d_fov_ != 0.00) {
        desc << (first ? "=" : ":") << "d_fov=" << d_fov_;
        first = false;
    }
    if (h_flip_ != false) {
        desc << (first ? "=" : ":") << "h_flip=" << (h_flip_ ? "1" : "0");
        first = false;
    }
    if (v_flip_ != false) {
        desc << (first ? "=" : ":") << "v_flip=" << (v_flip_ ? "1" : "0");
        first = false;
    }
    if (d_flip_ != false) {
        desc << (first ? "=" : ":") << "d_flip=" << (d_flip_ ? "1" : "0");
        first = false;
    }
    if (ih_flip_ != false) {
        desc << (first ? "=" : ":") << "ih_flip=" << (ih_flip_ ? "1" : "0");
        first = false;
    }
    if (iv_flip_ != false) {
        desc << (first ? "=" : ":") << "iv_flip=" << (iv_flip_ ? "1" : "0");
        first = false;
    }
    if (in_trans_ != false) {
        desc << (first ? "=" : ":") << "in_trans=" << (in_trans_ ? "1" : "0");
        first = false;
    }
    if (out_trans_ != false) {
        desc << (first ? "=" : ":") << "out_trans=" << (out_trans_ ? "1" : "0");
        first = false;
    }
    if (ih_fov_ != 0.00) {
        desc << (first ? "=" : ":") << "ih_fov=" << ih_fov_;
        first = false;
    }
    if (iv_fov_ != 0.00) {
        desc << (first ? "=" : ":") << "iv_fov=" << iv_fov_;
        first = false;
    }
    if (id_fov_ != 0.00) {
        desc << (first ? "=" : ":") << "id_fov=" << id_fov_;
        first = false;
    }
    if (h_offset_ != 0.00) {
        desc << (first ? "=" : ":") << "h_offset=" << h_offset_;
        first = false;
    }
    if (v_offset_ != 0.00) {
        desc << (first ? "=" : ":") << "v_offset=" << v_offset_;
        first = false;
    }
    if (alpha_mask_ != false) {
        desc << (first ? "=" : ":") << "alpha_mask=" << (alpha_mask_ ? "1" : "0");
        first = false;
    }
    if (reset_rot_ != false) {
        desc << (first ? "=" : ":") << "reset_rot=" << (reset_rot_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
