#include "Signalstats_bindings.hpp"

namespace py = pybind11;

void bind_Signalstats(py::module_ &m) {
    py::class_<Signalstats, FilterBase, std::shared_ptr<Signalstats>>(m, "Signalstats")
        .def(py::init<int, int, std::string>(),
             py::arg("stat") = 0,
             py::arg("out") = -1,
             py::arg("color") = "yellow")
        .def("setStat", &Signalstats::setStat)
        .def("getStat", &Signalstats::getStat)
        .def("setOut", &Signalstats::setOut)
        .def("getOut", &Signalstats::getOut)
        .def("setColor", &Signalstats::setColor)
        .def("getColor", &Signalstats::getColor)
        ;
}