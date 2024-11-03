#include "Tpad_bindings.hpp"

namespace py = pybind11;

void bind_Tpad(py::module_ &m) {
    py::class_<Tpad, FilterBase, std::shared_ptr<Tpad>>(m, "Tpad")
        .def(py::init<int, int, int, int, int64_t, int64_t, std::string>(),
             py::arg("start") = 0,
             py::arg("stop") = 0,
             py::arg("start_mode") = 0,
             py::arg("stop_mode") = 0,
             py::arg("start_duration") = 0ULL,
             py::arg("stop_duration") = 0ULL,
             py::arg("color") = "black")
        .def("setStart", &Tpad::setStart)
        .def("getStart", &Tpad::getStart)
        .def("setStop", &Tpad::setStop)
        .def("getStop", &Tpad::getStop)
        .def("setStart_mode", &Tpad::setStart_mode)
        .def("getStart_mode", &Tpad::getStart_mode)
        .def("setStop_mode", &Tpad::setStop_mode)
        .def("getStop_mode", &Tpad::getStop_mode)
        .def("setStart_duration", &Tpad::setStart_duration)
        .def("getStart_duration", &Tpad::getStart_duration)
        .def("setStop_duration", &Tpad::setStop_duration)
        .def("getStop_duration", &Tpad::getStop_duration)
        .def("setColor", &Tpad::setColor)
        .def("getColor", &Tpad::getColor)
        ;
}