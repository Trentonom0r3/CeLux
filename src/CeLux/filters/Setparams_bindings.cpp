#include "Setparams_bindings.hpp"

namespace py = pybind11;

void bind_Setparams(py::module_ &m) {
    py::class_<Setparams, FilterBase, std::shared_ptr<Setparams>>(m, "Setparams")
        .def(py::init<int, int, int, int, int>(),
             py::arg("field_mode") = -1,
             py::arg("range") = -1,
             py::arg("color_primaries") = -1,
             py::arg("color_trc") = -1,
             py::arg("colorspace") = -1)
        .def("setField_mode", &Setparams::setField_mode)
        .def("getField_mode", &Setparams::getField_mode)
        .def("setRange", &Setparams::setRange)
        .def("getRange", &Setparams::getRange)
        .def("setColor_primaries", &Setparams::setColor_primaries)
        .def("getColor_primaries", &Setparams::getColor_primaries)
        .def("setColor_trc", &Setparams::setColor_trc)
        .def("getColor_trc", &Setparams::getColor_trc)
        .def("setColorspace", &Setparams::setColorspace)
        .def("getColorspace", &Setparams::getColorspace)
        ;
}