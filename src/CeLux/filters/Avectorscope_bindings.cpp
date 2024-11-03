#include "Avectorscope_bindings.hpp"

namespace py = pybind11;

void bind_Avectorscope(py::module_ &m) {
    py::class_<Avectorscope, FilterBase, std::shared_ptr<Avectorscope>>(m, "Avectorscope")
        .def(py::init<int, std::pair<int, int>, std::pair<int, int>, int, int, int, int, int, int, int, int, double, int, int, bool, int>(),
             py::arg("mode") = 0,
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rc") = 40,
             py::arg("gc") = 160,
             py::arg("bc") = 80,
             py::arg("ac") = 255,
             py::arg("rf") = 15,
             py::arg("gf") = 10,
             py::arg("bf") = 5,
             py::arg("af") = 5,
             py::arg("zoom") = 1.00,
             py::arg("draw") = 0,
             py::arg("scale") = 0,
             py::arg("swap") = true,
             py::arg("mirror") = 0)
        .def("setMode", &Avectorscope::setMode)
        .def("getMode", &Avectorscope::getMode)
        .def("setRate", &Avectorscope::setRate)
        .def("getRate", &Avectorscope::getRate)
        .def("setSize", &Avectorscope::setSize)
        .def("getSize", &Avectorscope::getSize)
        .def("setRc", &Avectorscope::setRc)
        .def("getRc", &Avectorscope::getRc)
        .def("setGc", &Avectorscope::setGc)
        .def("getGc", &Avectorscope::getGc)
        .def("setBc", &Avectorscope::setBc)
        .def("getBc", &Avectorscope::getBc)
        .def("setAc", &Avectorscope::setAc)
        .def("getAc", &Avectorscope::getAc)
        .def("setRf", &Avectorscope::setRf)
        .def("getRf", &Avectorscope::getRf)
        .def("setGf", &Avectorscope::setGf)
        .def("getGf", &Avectorscope::getGf)
        .def("setBf", &Avectorscope::setBf)
        .def("getBf", &Avectorscope::getBf)
        .def("setAf", &Avectorscope::setAf)
        .def("getAf", &Avectorscope::getAf)
        .def("setZoom", &Avectorscope::setZoom)
        .def("getZoom", &Avectorscope::getZoom)
        .def("setDraw", &Avectorscope::setDraw)
        .def("getDraw", &Avectorscope::getDraw)
        .def("setScale", &Avectorscope::setScale)
        .def("getScale", &Avectorscope::getScale)
        .def("setSwap", &Avectorscope::setSwap)
        .def("getSwap", &Avectorscope::getSwap)
        .def("setMirror", &Avectorscope::setMirror)
        .def("getMirror", &Avectorscope::getMirror)
        ;
}