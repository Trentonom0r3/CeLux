#include "filter_bindings_chunk_3.hpp"
// Include filter binding headers for this chunk
#include "Maskedthreshold_bindings.hpp"
#include "Maskfun_bindings.hpp"
#include "Median_bindings.hpp"
#include "Mergeplanes_bindings.hpp"
#include "Mestimate_bindings.hpp"
#include "Metadata_bindings.hpp"
#include "Midequalizer_bindings.hpp"
#include "Minterpolate_bindings.hpp"
#include "Mix_bindings.hpp"
#include "Monochrome_bindings.hpp"
#include "Morpho_bindings.hpp"
#include "Msad_bindings.hpp"
#include "Multiply_bindings.hpp"
#include "Negate_bindings.hpp"
#include "Nlmeans_bindings.hpp"
#include "Noformat_bindings.hpp"
#include "Noise_bindings.hpp"
#include "Normalize_bindings.hpp"
#include "Null_bindings.hpp"
#include "Oscilloscope_bindings.hpp"
#include "Overlay_bindings.hpp"
#include "Pad_bindings.hpp"
#include "Palettegen_bindings.hpp"
#include "Paletteuse_bindings.hpp"
#include "Photosensitivity_bindings.hpp"
#include "Pixdesctest_bindings.hpp"
#include "Pixelize_bindings.hpp"
#include "Pixscope_bindings.hpp"
#include "Premultiply_bindings.hpp"
#include "Prewitt_bindings.hpp"
#include "Pseudocolor_bindings.hpp"
#include "Psnr_bindings.hpp"
#include "Qp_bindings.hpp"
#include "Random_bindings.hpp"
#include "Readeia608_bindings.hpp"
#include "Readvitc_bindings.hpp"
#include "Remap_bindings.hpp"
#include "Removegrain_bindings.hpp"
#include "Removelogo_bindings.hpp"
#include "Reverse_bindings.hpp"
#include "Rgbashift_bindings.hpp"
#include "Roberts_bindings.hpp"
#include "Rotate_bindings.hpp"
#include "Scale_bindings.hpp"
#include "Scale2ref_bindings.hpp"
#include "Scdet_bindings.hpp"
#include "Scharr_bindings.hpp"
#include "Scroll_bindings.hpp"
#include "Segment_bindings.hpp"
#include "Select_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_3(py::module_ &m) {
    bind_Maskedthreshold(m);
    bind_Maskfun(m);
    bind_Median(m);
    bind_Mergeplanes(m);
    bind_Mestimate(m);
    bind_Metadata(m);
    bind_Midequalizer(m);
    bind_Minterpolate(m);
    bind_Mix(m);
    bind_Monochrome(m);
    bind_Morpho(m);
    bind_Msad(m);
    bind_Multiply(m);
    bind_Negate(m);
    bind_Nlmeans(m);
    bind_Noformat(m);
    bind_Noise(m);
    bind_Normalize(m);
    bind_Null(m);
    bind_Oscilloscope(m);
    bind_Overlay(m);
    bind_Pad(m);
    bind_Palettegen(m);
    bind_Paletteuse(m);
    bind_Photosensitivity(m);
    bind_Pixdesctest(m);
    bind_Pixelize(m);
    bind_Pixscope(m);
    bind_Premultiply(m);
    bind_Prewitt(m);
    bind_Pseudocolor(m);
    bind_Psnr(m);
    bind_Qp(m);
    bind_Random(m);
    bind_Readeia608(m);
    bind_Readvitc(m);
    bind_Remap(m);
    bind_Removegrain(m);
    bind_Removelogo(m);
    bind_Reverse(m);
    bind_Rgbashift(m);
    bind_Roberts(m);
    bind_Rotate(m);
    bind_Scale(m);
    bind_Scale2ref(m);
    bind_Scdet(m);
    bind_Scharr(m);
    bind_Scroll(m);
    bind_Segment(m);
    bind_Select(m);
}
