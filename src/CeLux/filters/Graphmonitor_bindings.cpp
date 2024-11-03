#include "Graphmonitor_bindings.hpp"

namespace py = pybind11;

void bind_Graphmonitor(py::module_ &m) {
    py::class_<Graphmonitor, FilterBase, std::shared_ptr<Graphmonitor>>(m, "Graphmonitor")
        .def(py::init<std::pair<int, int>, float, int, int, std::pair<int, int>>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("opacity") = 0.90,
             py::arg("mode") = 0,
             py::arg("flags") = 1,
             py::arg("rate") = std::make_pair<int, int>(0, 1))
        .def("setSize", &Graphmonitor::setSize)
        .def("getSize", &Graphmonitor::getSize)
        .def("setOpacity", &Graphmonitor::setOpacity)
        .def("getOpacity", &Graphmonitor::getOpacity)
        .def("setMode", &Graphmonitor::setMode)
        .def("getMode", &Graphmonitor::getMode)
        .def("setFlags", &Graphmonitor::setFlags)
        .def("getFlags", &Graphmonitor::getFlags)
        .def("setRate", &Graphmonitor::setRate)
        .def("getRate", &Graphmonitor::getRate)
        ;
}