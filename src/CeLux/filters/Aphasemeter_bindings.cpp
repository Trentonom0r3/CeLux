#include "Aphasemeter_bindings.hpp"

namespace py = pybind11;

void bind_Aphasemeter(py::module_ &m) {
    py::class_<Aphasemeter, FilterBase, std::shared_ptr<Aphasemeter>>(m, "Aphasemeter")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int, int, int, std::string, bool, bool, float, float, int64_t>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rc") = 2,
             py::arg("gc") = 7,
             py::arg("bc") = 1,
             py::arg("mpc") = "none",
             py::arg("video") = true,
             py::arg("phasing") = false,
             py::arg("tolerance") = 0.00,
             py::arg("angle") = 170.00,
             py::arg("duration") = 2000000ULL)
        .def("setRate", &Aphasemeter::setRate)
        .def("getRate", &Aphasemeter::getRate)
        .def("setSize", &Aphasemeter::setSize)
        .def("getSize", &Aphasemeter::getSize)
        .def("setRc", &Aphasemeter::setRc)
        .def("getRc", &Aphasemeter::getRc)
        .def("setGc", &Aphasemeter::setGc)
        .def("getGc", &Aphasemeter::getGc)
        .def("setBc", &Aphasemeter::setBc)
        .def("getBc", &Aphasemeter::getBc)
        .def("setMpc", &Aphasemeter::setMpc)
        .def("getMpc", &Aphasemeter::getMpc)
        .def("setVideo", &Aphasemeter::setVideo)
        .def("getVideo", &Aphasemeter::getVideo)
        .def("setPhasing", &Aphasemeter::setPhasing)
        .def("getPhasing", &Aphasemeter::getPhasing)
        .def("setTolerance", &Aphasemeter::setTolerance)
        .def("getTolerance", &Aphasemeter::getTolerance)
        .def("setAngle", &Aphasemeter::setAngle)
        .def("getAngle", &Aphasemeter::getAngle)
        .def("setDuration", &Aphasemeter::setDuration)
        .def("getDuration", &Aphasemeter::getDuration)
        ;
}