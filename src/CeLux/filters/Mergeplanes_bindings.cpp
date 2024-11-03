#include "Mergeplanes_bindings.hpp"

namespace py = pybind11;

void bind_Mergeplanes(py::module_ &m) {
    py::class_<Mergeplanes, FilterBase, std::shared_ptr<Mergeplanes>>(m, "Mergeplanes")
        .def(py::init<std::string, int, int, int, int, int, int, int, int>(),
             py::arg("format") = "yuva444p",
             py::arg("map0s") = 0,
             py::arg("map0p") = 0,
             py::arg("map1s") = 0,
             py::arg("map1p") = 0,
             py::arg("map2s") = 0,
             py::arg("map2p") = 0,
             py::arg("map3s") = 0,
             py::arg("map3p") = 0)
        .def("setFormat", &Mergeplanes::setFormat)
        .def("getFormat", &Mergeplanes::getFormat)
        .def("setMap0s", &Mergeplanes::setMap0s)
        .def("getMap0s", &Mergeplanes::getMap0s)
        .def("setMap0p", &Mergeplanes::setMap0p)
        .def("getMap0p", &Mergeplanes::getMap0p)
        .def("setMap1s", &Mergeplanes::setMap1s)
        .def("getMap1s", &Mergeplanes::getMap1s)
        .def("setMap1p", &Mergeplanes::setMap1p)
        .def("getMap1p", &Mergeplanes::getMap1p)
        .def("setMap2s", &Mergeplanes::setMap2s)
        .def("getMap2s", &Mergeplanes::getMap2s)
        .def("setMap2p", &Mergeplanes::setMap2p)
        .def("getMap2p", &Mergeplanes::getMap2p)
        .def("setMap3s", &Mergeplanes::setMap3s)
        .def("getMap3s", &Mergeplanes::getMap3s)
        .def("setMap3p", &Mergeplanes::setMap3p)
        .def("getMap3p", &Mergeplanes::getMap3p)
        ;
}