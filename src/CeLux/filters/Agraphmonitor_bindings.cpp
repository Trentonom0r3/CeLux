#include "Agraphmonitor_bindings.hpp"

namespace py = pybind11;

void bind_Agraphmonitor(py::module_ &m) {
    py::class_<Agraphmonitor, FilterBase, std::shared_ptr<Agraphmonitor>>(m, "Agraphmonitor")
        .def(py::init<std::pair<int, int>, float, int, int, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("opacity") = 0.90,
             py::arg("mode") = 0,
             py::arg("flags") = 1,
             py::arg("rate") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Agraphmonitor::setSize)
        .def("getSize", &Agraphmonitor::getSize)
        .def("setOpacity", &Agraphmonitor::setOpacity)
        .def("getOpacity", &Agraphmonitor::getOpacity)
        .def("setMode", &Agraphmonitor::setMode)
        .def("getMode", &Agraphmonitor::getMode)
        .def("setFlags", &Agraphmonitor::setFlags)
        .def("getFlags", &Agraphmonitor::getFlags)
        .def("setRate", &Agraphmonitor::setRate)
        .def("getRate", &Agraphmonitor::getRate)
        ;
}