#include "filter_bindings_chunk_1.hpp"
// Include filter binding headers for this chunk
#include "Convolution_bindings.hpp"
#include "Convolve_bindings.hpp"
#include "Copy_bindings.hpp"
#include "Corr_bindings.hpp"
#include "Crop_bindings.hpp"
#include "Curves_bindings.hpp"
#include "Datascope_bindings.hpp"
#include "Dblur_bindings.hpp"
#include "Dctdnoiz_bindings.hpp"
#include "Deband_bindings.hpp"
#include "Deblock_bindings.hpp"
#include "Decimate_bindings.hpp"
#include "Deconvolve_bindings.hpp"
#include "Dedot_bindings.hpp"
#include "Deflate_bindings.hpp"
#include "Deflicker_bindings.hpp"
#include "Dejudder_bindings.hpp"
#include "Derain_bindings.hpp"
#include "Deshake_bindings.hpp"
#include "Despill_bindings.hpp"
#include "Detelecine_bindings.hpp"
#include "Dilation_bindings.hpp"
#include "Displace_bindings.hpp"
#include "Dnn_classify_bindings.hpp"
#include "Dnn_detect_bindings.hpp"
#include "Dnn_processing_bindings.hpp"
#include "Doubleweave_bindings.hpp"
#include "Drawbox_bindings.hpp"
#include "Drawgraph_bindings.hpp"
#include "Drawgrid_bindings.hpp"
#include "Edgedetect_bindings.hpp"
#include "Elbg_bindings.hpp"
#include "Entropy_bindings.hpp"
#include "Epx_bindings.hpp"
#include "Erosion_bindings.hpp"
#include "Estdif_bindings.hpp"
#include "Exposure_bindings.hpp"
#include "Extractplanes_bindings.hpp"
#include "Fade_bindings.hpp"
#include "Feedback_bindings.hpp"
#include "Fftdnoiz_bindings.hpp"
#include "Fftfilt_bindings.hpp"
#include "Field_bindings.hpp"
#include "Fieldhint_bindings.hpp"
#include "Fieldmatch_bindings.hpp"
#include "Fieldorder_bindings.hpp"
#include "Fillborders_bindings.hpp"
#include "Floodfill_bindings.hpp"
#include "Format_bindings.hpp"
#include "Fps_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_1(py::module_ &m) {
    bind_Convolution(m);
    bind_Convolve(m);
    bind_Copy(m);
    bind_Corr(m);
    bind_Crop(m);
    bind_Curves(m);
    bind_Datascope(m);
    bind_Dblur(m);
    bind_Dctdnoiz(m);
    bind_Deband(m);
    bind_Deblock(m);
    bind_Decimate(m);
    bind_Deconvolve(m);
    bind_Dedot(m);
    bind_Deflate(m);
    bind_Deflicker(m);
    bind_Dejudder(m);
    bind_Derain(m);
    bind_Deshake(m);
    bind_Despill(m);
    bind_Detelecine(m);
    bind_Dilation(m);
    bind_Displace(m);
    bind_Dnn_classify(m);
    bind_Dnn_detect(m);
    bind_Dnn_processing(m);
    bind_Doubleweave(m);
    bind_Drawbox(m);
    bind_Drawgraph(m);
    bind_Drawgrid(m);
    bind_Edgedetect(m);
    bind_Elbg(m);
    bind_Entropy(m);
    bind_Epx(m);
    bind_Erosion(m);
    bind_Estdif(m);
    bind_Exposure(m);
    bind_Extractplanes(m);
    bind_Fade(m);
    bind_Feedback(m);
    bind_Fftdnoiz(m);
    bind_Fftfilt(m);
    bind_Field(m);
    bind_Fieldhint(m);
    bind_Fieldmatch(m);
    bind_Fieldorder(m);
    bind_Fillborders(m);
    bind_Floodfill(m);
    bind_Format(m);
    bind_Fps(m);
}
