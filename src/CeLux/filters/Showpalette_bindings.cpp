#include "Showpalette_bindings.hpp"

namespace py = pybind11;

void bind_Showpalette(py::module_ &m) {
    py::class_<Showpalette, FilterBase, std::shared_ptr<Showpalette>>(m, "Showpalette")
        .def(py::init<int>(),
             py::arg("pixelBoxSize") = 30)
        .def("setPixelBoxSize", &Showpalette::setPixelBoxSize)
        .def("getPixelBoxSize", &Showpalette::getPixelBoxSize)
        ;
}