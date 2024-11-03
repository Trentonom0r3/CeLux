#include "filter_bindings_chunk_0.hpp"
// Include filter binding headers for this chunk
#include "Acopy_bindings.hpp"
#include "Aderivative_bindings.hpp"
#include "Aintegral_bindings.hpp"
#include "Alatency_bindings.hpp"
#include "Amultiply_bindings.hpp"
#include "Anull_bindings.hpp"
#include "Apsnr_bindings.hpp"
#include "Areverse_bindings.hpp"
#include "Asdr_bindings.hpp"
#include "Ashowinfo_bindings.hpp"
#include "Asisdr_bindings.hpp"
#include "Earwax_bindings.hpp"
#include "Volumedetect_bindings.hpp"
#include "Anullsink_bindings.hpp"
#include "Addroi_bindings.hpp"
#include "Alphaextract_bindings.hpp"
#include "Alphamerge_bindings.hpp"
#include "Amplify_bindings.hpp"
#include "Atadenoise_bindings.hpp"
#include "Avgblur_bindings.hpp"
#include "Backgroundkey_bindings.hpp"
#include "Bbox_bindings.hpp"
#include "Bench_bindings.hpp"
#include "Bilateral_bindings.hpp"
#include "Bitplanenoise_bindings.hpp"
#include "Blackdetect_bindings.hpp"
#include "Blend_bindings.hpp"
#include "Blockdetect_bindings.hpp"
#include "Blurdetect_bindings.hpp"
#include "Bm3d_bindings.hpp"
#include "Bwdif_bindings.hpp"
#include "Cas_bindings.hpp"
#include "Ccrepack_bindings.hpp"
#include "Chromahold_bindings.hpp"
#include "Chromakey_bindings.hpp"
#include "Chromanr_bindings.hpp"
#include "Chromashift_bindings.hpp"
#include "Ciescope_bindings.hpp"
#include "Codecview_bindings.hpp"
#include "Colorbalance_bindings.hpp"
#include "Colorchannelmixer_bindings.hpp"
#include "Colorcontrast_bindings.hpp"
#include "Colorcorrect_bindings.hpp"
#include "Colorize_bindings.hpp"
#include "Colorkey_bindings.hpp"
#include "Colorhold_bindings.hpp"
#include "Colorlevels_bindings.hpp"
#include "Colormap_bindings.hpp"
#include "Colorspace_bindings.hpp"
#include "Colortemperature_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_0(py::module_ &m) {
    bind_Acopy(m);
    bind_Aderivative(m);
    bind_Aintegral(m);
    bind_Alatency(m);
    bind_Amultiply(m);
    bind_Anull(m);
    bind_Apsnr(m);
    bind_Areverse(m);
    bind_Asdr(m);
    bind_Ashowinfo(m);
    bind_Asisdr(m);
    bind_Earwax(m);
    bind_Volumedetect(m);
    bind_Anullsink(m);
    bind_Addroi(m);
    bind_Alphaextract(m);
    bind_Alphamerge(m);
    bind_Amplify(m);
    bind_Atadenoise(m);
    bind_Avgblur(m);
    bind_Backgroundkey(m);
    bind_Bbox(m);
    bind_Bench(m);
    bind_Bilateral(m);
    bind_Bitplanenoise(m);
    bind_Blackdetect(m);
    bind_Blend(m);
    bind_Blockdetect(m);
    bind_Blurdetect(m);
    bind_Bm3d(m);
    bind_Bwdif(m);
    bind_Cas(m);
    bind_Ccrepack(m);
    bind_Chromahold(m);
    bind_Chromakey(m);
    bind_Chromanr(m);
    bind_Chromashift(m);
    bind_Ciescope(m);
    bind_Codecview(m);
    bind_Colorbalance(m);
    bind_Colorchannelmixer(m);
    bind_Colorcontrast(m);
    bind_Colorcorrect(m);
    bind_Colorize(m);
    bind_Colorkey(m);
    bind_Colorhold(m);
    bind_Colorlevels(m);
    bind_Colormap(m);
    bind_Colorspace(m);
    bind_Colortemperature(m);
}
