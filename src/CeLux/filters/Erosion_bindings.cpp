#include "Erosion_bindings.hpp"

namespace py = pybind11;

void bind_Erosion(py::module_ &m) {
    py::class_<Erosion, FilterBase, std::shared_ptr<Erosion>>(m, "Erosion")
        .def(py::init<int, int, int, int, int>(),
             py::arg("coordinates") = 255,
             py::arg("threshold0") = 65535,
             py::arg("threshold1") = 65535,
             py::arg("threshold2") = 65535,
             py::arg("threshold3") = 65535)
        .def("setCoordinates", &Erosion::setCoordinates)
        .def("getCoordinates", &Erosion::getCoordinates)
        .def("setThreshold0", &Erosion::setThreshold0)
        .def("getThreshold0", &Erosion::getThreshold0)
        .def("setThreshold1", &Erosion::setThreshold1)
        .def("getThreshold1", &Erosion::getThreshold1)
        .def("setThreshold2", &Erosion::setThreshold2)
        .def("getThreshold2", &Erosion::getThreshold2)
        .def("setThreshold3", &Erosion::setThreshold3)
        .def("getThreshold3", &Erosion::getThreshold3)
        ;
}