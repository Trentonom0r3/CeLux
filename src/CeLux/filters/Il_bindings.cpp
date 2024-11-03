#include "Il_bindings.hpp"

namespace py = pybind11;

void bind_Il(py::module_ &m) {
    py::class_<Il, FilterBase, std::shared_ptr<Il>>(m, "Il")
        .def(py::init<int, int, int, bool, bool, bool>(),
             py::arg("luma_mode") = 0,
             py::arg("chroma_mode") = 0,
             py::arg("alpha_mode") = 0,
             py::arg("luma_swap") = false,
             py::arg("chroma_swap") = false,
             py::arg("alpha_swap") = false)
        .def("setLuma_mode", &Il::setLuma_mode)
        .def("getLuma_mode", &Il::getLuma_mode)
        .def("setChroma_mode", &Il::setChroma_mode)
        .def("getChroma_mode", &Il::getChroma_mode)
        .def("setAlpha_mode", &Il::setAlpha_mode)
        .def("getAlpha_mode", &Il::getAlpha_mode)
        .def("setLuma_swap", &Il::setLuma_swap)
        .def("getLuma_swap", &Il::getLuma_swap)
        .def("setChroma_swap", &Il::setChroma_swap)
        .def("getChroma_swap", &Il::getChroma_swap)
        .def("setAlpha_swap", &Il::setAlpha_swap)
        .def("getAlpha_swap", &Il::getAlpha_swap)
        ;
}