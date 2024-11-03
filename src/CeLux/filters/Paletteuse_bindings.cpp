#include "Paletteuse_bindings.hpp"

namespace py = pybind11;

void bind_Paletteuse(py::module_ &m) {
    py::class_<Paletteuse, FilterBase, std::shared_ptr<Paletteuse>>(m, "Paletteuse")
        .def(py::init<int, int, int, bool, int, std::string>(),
             py::arg("dither") = 5,
             py::arg("bayer_scale") = 2,
             py::arg("diff_mode") = 0,
             py::arg("new_") = false,
             py::arg("alpha_threshold") = 128,
             py::arg("debug_kdtree") = "")
        .def("setDither", &Paletteuse::setDither)
        .def("getDither", &Paletteuse::getDither)
        .def("setBayer_scale", &Paletteuse::setBayer_scale)
        .def("getBayer_scale", &Paletteuse::getBayer_scale)
        .def("setDiff_mode", &Paletteuse::setDiff_mode)
        .def("getDiff_mode", &Paletteuse::getDiff_mode)
        .def("setNew_", &Paletteuse::setNew_)
        .def("getNew_", &Paletteuse::getNew_)
        .def("setAlpha_threshold", &Paletteuse::setAlpha_threshold)
        .def("getAlpha_threshold", &Paletteuse::getAlpha_threshold)
        .def("setDebug_kdtree", &Paletteuse::setDebug_kdtree)
        .def("getDebug_kdtree", &Paletteuse::getDebug_kdtree)
        ;
}