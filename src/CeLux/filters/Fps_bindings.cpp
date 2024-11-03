#include "Fps_bindings.hpp"

namespace py = pybind11;

void bind_Fps(py::module_ &m) {
    py::class_<Fps, FilterBase, std::shared_ptr<Fps>>(m, "Fps")
        .def(py::init<std::string, double, int, int>(),
             py::arg("fps") = "25",
             py::arg("start_time") = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00,
             py::arg("round") = 5,
             py::arg("eof_action") = 0)
        .def("setFps", &Fps::setFps)
        .def("getFps", &Fps::getFps)
        .def("setStart_time", &Fps::setStart_time)
        .def("getStart_time", &Fps::getStart_time)
        .def("setRound", &Fps::setRound)
        .def("getRound", &Fps::getRound)
        .def("setEof_action", &Fps::setEof_action)
        .def("getEof_action", &Fps::getEof_action)
        ;
}