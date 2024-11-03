#include "filter_bindings_chunk_4.hpp"
// Include filter binding headers for this chunk
#include "Selectivecolor_bindings.hpp"
#include "Separatefields_bindings.hpp"
#include "Setdar_bindings.hpp"
#include "Setfield_bindings.hpp"
#include "Setparams_bindings.hpp"
#include "Setpts_bindings.hpp"
#include "Setrange_bindings.hpp"
#include "Setsar_bindings.hpp"
#include "Settb_bindings.hpp"
#include "Shear_bindings.hpp"
#include "Showinfo_bindings.hpp"
#include "Showpalette_bindings.hpp"
#include "Shuffleframes_bindings.hpp"
#include "Shufflepixels_bindings.hpp"
#include "Shuffleplanes_bindings.hpp"
#include "Sidedata_bindings.hpp"
#include "Signalstats_bindings.hpp"
#include "Siti_bindings.hpp"
#include "Sobel_bindings.hpp"
#include "Sr_bindings.hpp"
#include "Ssim_bindings.hpp"
#include "Ssim360_bindings.hpp"
#include "Swaprect_bindings.hpp"
#include "Swapuv_bindings.hpp"
#include "Tblend_bindings.hpp"
#include "Telecine_bindings.hpp"
#include "Thistogram_bindings.hpp"
#include "Threshold_bindings.hpp"
#include "Thumbnail_bindings.hpp"
#include "Tile_bindings.hpp"
#include "Tiltandshift_bindings.hpp"
#include "Tlut2_bindings.hpp"
#include "Tmedian_bindings.hpp"
#include "Tmidequalizer_bindings.hpp"
#include "Tmix_bindings.hpp"
#include "Tonemap_bindings.hpp"
#include "Tpad_bindings.hpp"
#include "Transpose_bindings.hpp"
#include "Trim_bindings.hpp"
#include "Unpremultiply_bindings.hpp"
#include "Unsharp_bindings.hpp"
#include "Untile_bindings.hpp"
#include "V360_bindings.hpp"
#include "Varblur_bindings.hpp"
#include "Vectorscope_bindings.hpp"
#include "Vflip_bindings.hpp"
#include "Vfrdet_bindings.hpp"
#include "Vibrance_bindings.hpp"
#include "Vif_bindings.hpp"
#include "Vignette_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_4(py::module_ &m) {
    bind_Selectivecolor(m);
    bind_Separatefields(m);
    bind_Setdar(m);
    bind_Setfield(m);
    bind_Setparams(m);
    bind_Setpts(m);
    bind_Setrange(m);
    bind_Setsar(m);
    bind_Settb(m);
    bind_Shear(m);
    bind_Showinfo(m);
    bind_Showpalette(m);
    bind_Shuffleframes(m);
    bind_Shufflepixels(m);
    bind_Shuffleplanes(m);
    bind_Sidedata(m);
    bind_Signalstats(m);
    bind_Siti(m);
    bind_Sobel(m);
    bind_Sr(m);
    bind_Ssim(m);
    bind_Ssim360(m);
    bind_Swaprect(m);
    bind_Swapuv(m);
    bind_Tblend(m);
    bind_Telecine(m);
    bind_Thistogram(m);
    bind_Threshold(m);
    bind_Thumbnail(m);
    bind_Tile(m);
    bind_Tiltandshift(m);
    bind_Tlut2(m);
    bind_Tmedian(m);
    bind_Tmidequalizer(m);
    bind_Tmix(m);
    bind_Tonemap(m);
    bind_Tpad(m);
    bind_Transpose(m);
    bind_Trim(m);
    bind_Unpremultiply(m);
    bind_Unsharp(m);
    bind_Untile(m);
    bind_V360(m);
    bind_Varblur(m);
    bind_Vectorscope(m);
    bind_Vflip(m);
    bind_Vfrdet(m);
    bind_Vibrance(m);
    bind_Vif(m);
    bind_Vignette(m);
}
