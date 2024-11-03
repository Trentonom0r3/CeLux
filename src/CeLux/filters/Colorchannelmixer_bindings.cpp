#include "Colorchannelmixer_bindings.hpp"

namespace py = pybind11;

void bind_Colorchannelmixer(py::module_ &m) {
    py::class_<Colorchannelmixer, FilterBase, std::shared_ptr<Colorchannelmixer>>(m, "Colorchannelmixer")
        .def(py::init<double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, int, double>(),
             py::arg("rr") = 1.00,
             py::arg("rg") = 0.00,
             py::arg("rb") = 0.00,
             py::arg("ra") = 0.00,
             py::arg("gr") = 0.00,
             py::arg("gg") = 1.00,
             py::arg("gb") = 0.00,
             py::arg("ga") = 0.00,
             py::arg("br") = 0.00,
             py::arg("bg") = 0.00,
             py::arg("bb") = 1.00,
             py::arg("ba") = 0.00,
             py::arg("ar") = 0.00,
             py::arg("ag") = 0.00,
             py::arg("ab") = 0.00,
             py::arg("aa") = 1.00,
             py::arg("pc") = 0,
             py::arg("pa") = 0.00)
        .def("setRr", &Colorchannelmixer::setRr)
        .def("getRr", &Colorchannelmixer::getRr)
        .def("setRg", &Colorchannelmixer::setRg)
        .def("getRg", &Colorchannelmixer::getRg)
        .def("setRb", &Colorchannelmixer::setRb)
        .def("getRb", &Colorchannelmixer::getRb)
        .def("setRa", &Colorchannelmixer::setRa)
        .def("getRa", &Colorchannelmixer::getRa)
        .def("setGr", &Colorchannelmixer::setGr)
        .def("getGr", &Colorchannelmixer::getGr)
        .def("setGg", &Colorchannelmixer::setGg)
        .def("getGg", &Colorchannelmixer::getGg)
        .def("setGb", &Colorchannelmixer::setGb)
        .def("getGb", &Colorchannelmixer::getGb)
        .def("setGa", &Colorchannelmixer::setGa)
        .def("getGa", &Colorchannelmixer::getGa)
        .def("setBr", &Colorchannelmixer::setBr)
        .def("getBr", &Colorchannelmixer::getBr)
        .def("setBg", &Colorchannelmixer::setBg)
        .def("getBg", &Colorchannelmixer::getBg)
        .def("setBb", &Colorchannelmixer::setBb)
        .def("getBb", &Colorchannelmixer::getBb)
        .def("setBa", &Colorchannelmixer::setBa)
        .def("getBa", &Colorchannelmixer::getBa)
        .def("setAr", &Colorchannelmixer::setAr)
        .def("getAr", &Colorchannelmixer::getAr)
        .def("setAg", &Colorchannelmixer::setAg)
        .def("getAg", &Colorchannelmixer::getAg)
        .def("setAb", &Colorchannelmixer::setAb)
        .def("getAb", &Colorchannelmixer::getAb)
        .def("setAa", &Colorchannelmixer::setAa)
        .def("getAa", &Colorchannelmixer::getAa)
        .def("setPc", &Colorchannelmixer::setPc)
        .def("getPc", &Colorchannelmixer::getPc)
        .def("setPa", &Colorchannelmixer::setPa)
        .def("getPa", &Colorchannelmixer::getPa)
        ;
}