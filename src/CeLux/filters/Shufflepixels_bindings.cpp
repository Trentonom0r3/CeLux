#include "Shufflepixels_bindings.hpp"

namespace py = pybind11;

void bind_Shufflepixels(py::module_ &m) {
    py::class_<Shufflepixels, FilterBase, std::shared_ptr<Shufflepixels>>(m, "Shufflepixels")
        .def(py::init<int, int, int, int, int64_t>(),
             py::arg("direction") = 0,
             py::arg("mode") = 0,
             py::arg("width") = 10,
             py::arg("height") = 10,
             py::arg("seed") = 0)
        .def("setDirection", &Shufflepixels::setDirection)
        .def("getDirection", &Shufflepixels::getDirection)
        .def("setMode", &Shufflepixels::setMode)
        .def("getMode", &Shufflepixels::getMode)
        .def("setWidth", &Shufflepixels::setWidth)
        .def("getWidth", &Shufflepixels::getWidth)
        .def("setHeight", &Shufflepixels::setHeight)
        .def("getHeight", &Shufflepixels::getHeight)
        .def("setSeed", &Shufflepixels::setSeed)
        .def("getSeed", &Shufflepixels::getSeed)
        ;
}