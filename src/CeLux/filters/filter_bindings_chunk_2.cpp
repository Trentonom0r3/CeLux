#include "filter_bindings_chunk_2.hpp"
// Include filter binding headers for this chunk
#include "Framepack_bindings.hpp"
#include "Framerate_bindings.hpp"
#include "Framestep_bindings.hpp"
#include "Freezedetect_bindings.hpp"
#include "Freezeframes_bindings.hpp"
#include "Fsync_bindings.hpp"
#include "Gblur_bindings.hpp"
#include "Geq_bindings.hpp"
#include "Gradfun_bindings.hpp"
#include "Graphmonitor_bindings.hpp"
#include "Grayworld_bindings.hpp"
#include "Greyedge_bindings.hpp"
#include "Guided_bindings.hpp"
#include "Haldclut_bindings.hpp"
#include "Hflip_bindings.hpp"
#include "Histogram_bindings.hpp"
#include "Hqx_bindings.hpp"
#include "Hstack_bindings.hpp"
#include "Hsvhold_bindings.hpp"
#include "Hsvkey_bindings.hpp"
#include "Hue_bindings.hpp"
#include "Huesaturation_bindings.hpp"
#include "Hwdownload_bindings.hpp"
#include "Hwmap_bindings.hpp"
#include "Hwupload_bindings.hpp"
#include "Hwupload_cuda_bindings.hpp"
#include "Hysteresis_bindings.hpp"
#include "Identity_bindings.hpp"
#include "Idet_bindings.hpp"
#include "Il_bindings.hpp"
#include "Inflate_bindings.hpp"
#include "Interleave_bindings.hpp"
#include "Kirsch_bindings.hpp"
#include "Lagfun_bindings.hpp"
#include "Latency_bindings.hpp"
#include "Lenscorrection_bindings.hpp"
#include "Limitdiff_bindings.hpp"
#include "Limiter_bindings.hpp"
#include "Loop_bindings.hpp"
#include "Lumakey_bindings.hpp"
#include "Lut_bindings.hpp"
#include "Lut1d_bindings.hpp"
#include "Lut2_bindings.hpp"
#include "Lut3d_bindings.hpp"
#include "Lutrgb_bindings.hpp"
#include "Lutyuv_bindings.hpp"
#include "Maskedclamp_bindings.hpp"
#include "Maskedmax_bindings.hpp"
#include "Maskedmerge_bindings.hpp"
#include "Maskedmin_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_2(py::module_ &m) {
    bind_Framepack(m);
    bind_Framerate(m);
    bind_Framestep(m);
    bind_Freezedetect(m);
    bind_Freezeframes(m);
    bind_Fsync(m);
    bind_Gblur(m);
    bind_Geq(m);
    bind_Gradfun(m);
    bind_Graphmonitor(m);
    bind_Grayworld(m);
    bind_Greyedge(m);
    bind_Guided(m);
    bind_Haldclut(m);
    bind_Hflip(m);
    bind_Histogram(m);
    bind_Hqx(m);
    bind_Hstack(m);
    bind_Hsvhold(m);
    bind_Hsvkey(m);
    bind_Hue(m);
    bind_Huesaturation(m);
    bind_Hwdownload(m);
    bind_Hwmap(m);
    bind_Hwupload(m);
    bind_Hwupload_cuda(m);
    bind_Hysteresis(m);
    bind_Identity(m);
    bind_Idet(m);
    bind_Il(m);
    bind_Inflate(m);
    bind_Interleave(m);
    bind_Kirsch(m);
    bind_Lagfun(m);
    bind_Latency(m);
    bind_Lenscorrection(m);
    bind_Limitdiff(m);
    bind_Limiter(m);
    bind_Loop(m);
    bind_Lumakey(m);
    bind_Lut(m);
    bind_Lut1d(m);
    bind_Lut2(m);
    bind_Lut3d(m);
    bind_Lutrgb(m);
    bind_Lutyuv(m);
    bind_Maskedclamp(m);
    bind_Maskedmax(m);
    bind_Maskedmerge(m);
    bind_Maskedmin(m);
}
