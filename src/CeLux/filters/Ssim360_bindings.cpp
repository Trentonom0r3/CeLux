#include "Ssim360_bindings.hpp"

namespace py = pybind11;

void bind_Ssim360(py::module_ &m) {
    py::class_<Ssim360, FilterBase, std::shared_ptr<Ssim360>>(m, "Ssim360")
        .def(py::init<std::string, int, int, int, int, int, int, float, float, int, std::string, int, int>(),
             py::arg("stats_file") = "",
             py::arg("compute_chroma") = 1,
             py::arg("frame_skip_ratio") = 0,
             py::arg("ref_projection") = 4,
             py::arg("main_projection") = 5,
             py::arg("ref_stereo") = 2,
             py::arg("main_stereo") = 3,
             py::arg("ref_pad") = 0.00,
             py::arg("main_pad") = 0.00,
             py::arg("use_tape") = 0,
             py::arg("heatmap_str") = "",
             py::arg("default_heatmap_width") = 32,
             py::arg("default_heatmap_height") = 16)
        .def("setStats_file", &Ssim360::setStats_file)
        .def("getStats_file", &Ssim360::getStats_file)
        .def("setCompute_chroma", &Ssim360::setCompute_chroma)
        .def("getCompute_chroma", &Ssim360::getCompute_chroma)
        .def("setFrame_skip_ratio", &Ssim360::setFrame_skip_ratio)
        .def("getFrame_skip_ratio", &Ssim360::getFrame_skip_ratio)
        .def("setRef_projection", &Ssim360::setRef_projection)
        .def("getRef_projection", &Ssim360::getRef_projection)
        .def("setMain_projection", &Ssim360::setMain_projection)
        .def("getMain_projection", &Ssim360::getMain_projection)
        .def("setRef_stereo", &Ssim360::setRef_stereo)
        .def("getRef_stereo", &Ssim360::getRef_stereo)
        .def("setMain_stereo", &Ssim360::setMain_stereo)
        .def("getMain_stereo", &Ssim360::getMain_stereo)
        .def("setRef_pad", &Ssim360::setRef_pad)
        .def("getRef_pad", &Ssim360::getRef_pad)
        .def("setMain_pad", &Ssim360::setMain_pad)
        .def("getMain_pad", &Ssim360::getMain_pad)
        .def("setUse_tape", &Ssim360::setUse_tape)
        .def("getUse_tape", &Ssim360::getUse_tape)
        .def("setHeatmap_str", &Ssim360::setHeatmap_str)
        .def("getHeatmap_str", &Ssim360::getHeatmap_str)
        .def("setDefault_heatmap_width", &Ssim360::setDefault_heatmap_width)
        .def("getDefault_heatmap_width", &Ssim360::getDefault_heatmap_width)
        .def("setDefault_heatmap_height", &Ssim360::setDefault_heatmap_height)
        .def("getDefault_heatmap_height", &Ssim360::getDefault_heatmap_height)
        ;
}