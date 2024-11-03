#include "Random_bindings.hpp"

namespace py = pybind11;

void bind_Random(py::module_ &m) {
    py::class_<Random, FilterBase, std::shared_ptr<Random>>(m, "Random")
        .def(py::init<int, int64_t>(),
             py::arg("frames") = 30,
             py::arg("seed") = 0)
        .def("setFrames", &Random::setFrames)
        .def("getFrames", &Random::getFrames)
        .def("setSeed", &Random::setSeed)
        .def("getSeed", &Random::getSeed)
        ;
}