#include "Dilation_bindings.hpp"

namespace py = pybind11;

void bind_Dilation(py::module_ &m) {
    py::class_<Dilation, FilterBase, std::shared_ptr<Dilation>>(m, "Dilation")
        .def(py::init<int, int, int, int, int>(),
             py::arg("coordinates") = 255,
             py::arg("threshold0") = 65535,
             py::arg("threshold1") = 65535,
             py::arg("threshold2") = 65535,
             py::arg("threshold3") = 65535)
        .def("setCoordinates", &Dilation::setCoordinates)
        .def("getCoordinates", &Dilation::getCoordinates)
        .def("setThreshold0", &Dilation::setThreshold0)
        .def("getThreshold0", &Dilation::getThreshold0)
        .def("setThreshold1", &Dilation::setThreshold1)
        .def("getThreshold1", &Dilation::getThreshold1)
        .def("setThreshold2", &Dilation::setThreshold2)
        .def("getThreshold2", &Dilation::getThreshold2)
        .def("setThreshold3", &Dilation::setThreshold3)
        .def("getThreshold3", &Dilation::getThreshold3)
        ;
}