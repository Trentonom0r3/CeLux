#include "Showcwt_bindings.hpp"

namespace py = pybind11;

void bind_Showcwt(py::module_ &m) {
    py::class_<Showcwt, FilterBase, std::shared_ptr<Showcwt>>(m, "Showcwt")
        .def(py::init<std::pair<int, int>, std::string, int, int, float, float, float, float, float, float, int, int, int, int, float, float>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = "25",
             py::arg("scale") = 0,
             py::arg("iscale") = 0,
             py::arg("min") = 20.00,
             py::arg("max") = 20000.00,
             py::arg("imin") = 0.00,
             py::arg("imax") = 1.00,
             py::arg("logb") = 0.00,
             py::arg("deviation") = 1.00,
             py::arg("pps") = 64,
             py::arg("mode") = 0,
             py::arg("slide") = 0,
             py::arg("direction") = 0,
             py::arg("bar") = 0.00,
             py::arg("rotation") = 0.00)
        .def("setSize", &Showcwt::setSize)
        .def("getSize", &Showcwt::getSize)
        .def("setRate", &Showcwt::setRate)
        .def("getRate", &Showcwt::getRate)
        .def("setScale", &Showcwt::setScale)
        .def("getScale", &Showcwt::getScale)
        .def("setIscale", &Showcwt::setIscale)
        .def("getIscale", &Showcwt::getIscale)
        .def("setMin", &Showcwt::setMin)
        .def("getMin", &Showcwt::getMin)
        .def("setMax", &Showcwt::setMax)
        .def("getMax", &Showcwt::getMax)
        .def("setImin", &Showcwt::setImin)
        .def("getImin", &Showcwt::getImin)
        .def("setImax", &Showcwt::setImax)
        .def("getImax", &Showcwt::getImax)
        .def("setLogb", &Showcwt::setLogb)
        .def("getLogb", &Showcwt::getLogb)
        .def("setDeviation", &Showcwt::setDeviation)
        .def("getDeviation", &Showcwt::getDeviation)
        .def("setPps", &Showcwt::setPps)
        .def("getPps", &Showcwt::getPps)
        .def("setMode", &Showcwt::setMode)
        .def("getMode", &Showcwt::getMode)
        .def("setSlide", &Showcwt::setSlide)
        .def("getSlide", &Showcwt::getSlide)
        .def("setDirection", &Showcwt::setDirection)
        .def("getDirection", &Showcwt::getDirection)
        .def("setBar", &Showcwt::setBar)
        .def("getBar", &Showcwt::getBar)
        .def("setRotation", &Showcwt::setRotation)
        .def("getRotation", &Showcwt::getRotation)
        ;
}