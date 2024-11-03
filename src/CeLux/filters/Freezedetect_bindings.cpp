#include "Freezedetect_bindings.hpp"

namespace py = pybind11;

void bind_Freezedetect(py::module_ &m) {
    py::class_<Freezedetect, FilterBase, std::shared_ptr<Freezedetect>>(m, "Freezedetect")
        .def(py::init<double, int64_t>(),
             py::arg("noise") = 0.00,
             py::arg("duration") = 2000000ULL)
        .def("setNoise", &Freezedetect::setNoise)
        .def("getNoise", &Freezedetect::getNoise)
        .def("setDuration", &Freezedetect::setDuration)
        .def("getDuration", &Freezedetect::getDuration)
        ;
}