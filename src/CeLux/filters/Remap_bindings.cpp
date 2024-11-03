#include "Remap_bindings.hpp"

namespace py = pybind11;

void bind_Remap(py::module_ &m) {
    py::class_<Remap, FilterBase, std::shared_ptr<Remap>>(m, "Remap")
        .def(py::init<int, std::string>(),
             py::arg("format") = 0,
             py::arg("fill") = "black")
        .def("setFormat", &Remap::setFormat)
        .def("getFormat", &Remap::getFormat)
        .def("setFill", &Remap::setFill)
        .def("getFill", &Remap::getFill)
        ;
}