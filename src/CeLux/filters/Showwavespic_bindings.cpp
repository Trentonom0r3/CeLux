#include "Showwavespic_bindings.hpp"

namespace py = pybind11;

void bind_Showwavespic(py::module_ &m) {
    py::class_<Showwavespic, FilterBase, std::shared_ptr<Showwavespic>>(m, "Showwavespic")
        .def(py::init<std::pair<int, int>, bool, std::string, int, int, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("split_channels") = false,
             py::arg("colors") = "red|green|blue|yellow|orange|lime|pink|magenta|brown",
             py::arg("scale") = 0,
             py::arg("draw") = 0,
             py::arg("filter") = 0)
        .def("setSize", &Showwavespic::setSize)
        .def("getSize", &Showwavespic::getSize)
        .def("setSplit_channels", &Showwavespic::setSplit_channels)
        .def("getSplit_channels", &Showwavespic::getSplit_channels)
        .def("setColors", &Showwavespic::setColors)
        .def("getColors", &Showwavespic::getColors)
        .def("setScale", &Showwavespic::setScale)
        .def("getScale", &Showwavespic::getScale)
        .def("setDraw", &Showwavespic::setDraw)
        .def("getDraw", &Showwavespic::getDraw)
        .def("setFilter", &Showwavespic::setFilter)
        .def("getFilter", &Showwavespic::getFilter)
        ;
}