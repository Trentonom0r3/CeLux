#include "filter_bindings_chunk_5.hpp"
// Include filter binding headers for this chunk
#include "Vmafmotion_bindings.hpp"
#include "Vstack_bindings.hpp"
#include "W3fdif_bindings.hpp"
#include "Waveform_bindings.hpp"
#include "Weave_bindings.hpp"
#include "Xbr_bindings.hpp"
#include "Xcorrelate_bindings.hpp"
#include "Xfade_bindings.hpp"
#include "Xmedian_bindings.hpp"
#include "Xstack_bindings.hpp"
#include "Yadif_bindings.hpp"
#include "Yaepblur_bindings.hpp"
#include "Zoompan_bindings.hpp"
#include "Allrgb_bindings.hpp"
#include "Allyuv_bindings.hpp"
#include "Cellauto_bindings.hpp"
#include "Color_bindings.hpp"
#include "Colorchart_bindings.hpp"
#include "Colorspectrum_bindings.hpp"
#include "Ddagrab_bindings.hpp"
#include "Gradients_bindings.hpp"
#include "Haldclutsrc_bindings.hpp"
#include "Life_bindings.hpp"
#include "Mandelbrot_bindings.hpp"
#include "Nullsrc_bindings.hpp"
#include "Pal75bars_bindings.hpp"
#include "Pal100bars_bindings.hpp"
#include "Rgbtestsrc_bindings.hpp"
#include "Sierpinski_bindings.hpp"
#include "Smptebars_bindings.hpp"
#include "Smptehdbars_bindings.hpp"
#include "Testsrc_bindings.hpp"
#include "Testsrc2_bindings.hpp"
#include "Yuvtestsrc_bindings.hpp"
#include "Zoneplate_bindings.hpp"
#include "Nullsink_bindings.hpp"
#include "A3dscope_bindings.hpp"
#include "Abitscope_bindings.hpp"
#include "Adrawgraph_bindings.hpp"
#include "Agraphmonitor_bindings.hpp"
#include "Ahistogram_bindings.hpp"
#include "Aphasemeter_bindings.hpp"
#include "Avectorscope_bindings.hpp"
#include "Showcqt_bindings.hpp"
#include "Showcwt_bindings.hpp"
#include "Showfreqs_bindings.hpp"
#include "Showspatial_bindings.hpp"
#include "Showspectrum_bindings.hpp"
#include "Showspectrumpic_bindings.hpp"
#include "Showvolume_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_5(py::module_ &m) {
    bind_Vmafmotion(m);
    bind_Vstack(m);
    bind_W3fdif(m);
    bind_Waveform(m);
    bind_Weave(m);
    bind_Xbr(m);
    bind_Xcorrelate(m);
    bind_Xfade(m);
    bind_Xmedian(m);
    bind_Xstack(m);
    bind_Yadif(m);
    bind_Yaepblur(m);
    bind_Zoompan(m);
    bind_Allrgb(m);
    bind_Allyuv(m);
    bind_Cellauto(m);
    bind_Color(m);
    bind_Colorchart(m);
    bind_Colorspectrum(m);
    bind_Ddagrab(m);
    bind_Gradients(m);
    bind_Haldclutsrc(m);
    bind_Life(m);
    bind_Mandelbrot(m);
    bind_Nullsrc(m);
    bind_Pal75bars(m);
    bind_Pal100bars(m);
    bind_Rgbtestsrc(m);
    bind_Sierpinski(m);
    bind_Smptebars(m);
    bind_Smptehdbars(m);
    bind_Testsrc(m);
    bind_Testsrc2(m);
    bind_Yuvtestsrc(m);
    bind_Zoneplate(m);
    bind_Nullsink(m);
    bind_A3dscope(m);
    bind_Abitscope(m);
    bind_Adrawgraph(m);
    bind_Agraphmonitor(m);
    bind_Ahistogram(m);
    bind_Aphasemeter(m);
    bind_Avectorscope(m);
    bind_Showcqt(m);
    bind_Showcwt(m);
    bind_Showfreqs(m);
    bind_Showspatial(m);
    bind_Showspectrum(m);
    bind_Showspectrumpic(m);
    bind_Showvolume(m);
}
