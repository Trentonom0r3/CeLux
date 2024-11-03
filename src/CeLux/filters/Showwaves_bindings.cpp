#include "Showwaves_bindings.hpp"

namespace py = pybind11;

void bind_Showwaves(py::module_ &m) {
    py::class_<Showwaves, FilterBase, std::shared_ptr<Showwaves>>(m, "Showwaves")
        .def(py::init<std::pair<int, int>, int, std::pair<int, int>, std::pair<int, int>, bool, std::string, int, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("mode") = 0,
             py::arg("howManySamplesToShowInTheSamePoint") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("split_channels") = false,
             py::arg("colors") = "red|green|blue|yellow|orange|lime|pink|magenta|brown",
             py::arg("scale") = 0,
             py::arg("draw") = 0)
        .def("setSize", &Showwaves::setSize)
        .def("getSize", &Showwaves::getSize)
        .def("setMode", &Showwaves::setMode)
        .def("getMode", &Showwaves::getMode)
        .def("setHowManySamplesToShowInTheSamePoint", &Showwaves::setHowManySamplesToShowInTheSamePoint)
        .def("getHowManySamplesToShowInTheSamePoint", &Showwaves::getHowManySamplesToShowInTheSamePoint)
        .def("setRate", &Showwaves::setRate)
        .def("getRate", &Showwaves::getRate)
        .def("setSplit_channels", &Showwaves::setSplit_channels)
        .def("getSplit_channels", &Showwaves::getSplit_channels)
        .def("setColors", &Showwaves::setColors)
        .def("getColors", &Showwaves::getColors)
        .def("setScale", &Showwaves::setScale)
        .def("getScale", &Showwaves::getScale)
        .def("setDraw", &Showwaves::setDraw)
        .def("getDraw", &Showwaves::getDraw)
        ;
}