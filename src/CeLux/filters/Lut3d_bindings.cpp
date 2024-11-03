#include "Lut3d_bindings.hpp"

namespace py = pybind11;

void bind_Lut3d(py::module_ &m) {
    py::class_<Lut3d, FilterBase, std::shared_ptr<Lut3d>>(m, "Lut3d")
        .def(py::init<std::string, int, int>(),
             py::arg("file") = "",
             py::arg("clut") = 1,
             py::arg("interp") = 2)
        .def("setFile", &Lut3d::setFile)
        .def("getFile", &Lut3d::getFile)
        .def("setClut", &Lut3d::setClut)
        .def("getClut", &Lut3d::getClut)
        .def("setInterp", &Lut3d::setInterp)
        .def("getInterp", &Lut3d::getInterp)
        ;
}