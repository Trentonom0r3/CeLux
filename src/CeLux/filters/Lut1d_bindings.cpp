#include "Lut1d_bindings.hpp"

namespace py = pybind11;

void bind_Lut1d(py::module_ &m) {
    py::class_<Lut1d, FilterBase, std::shared_ptr<Lut1d>>(m, "Lut1d")
        .def(py::init<std::string, int>(),
             py::arg("file") = "",
             py::arg("interp") = 1)
        .def("setFile", &Lut1d::setFile)
        .def("getFile", &Lut1d::getFile)
        .def("setInterp", &Lut1d::setInterp)
        .def("getInterp", &Lut1d::getInterp)
        ;
}