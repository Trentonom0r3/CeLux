#include "Shuffleplanes_bindings.hpp"

namespace py = pybind11;

void bind_Shuffleplanes(py::module_ &m) {
    py::class_<Shuffleplanes, FilterBase, std::shared_ptr<Shuffleplanes>>(m, "Shuffleplanes")
        .def(py::init<int, int, int, int>(),
             py::arg("map0") = 0,
             py::arg("map1") = 1,
             py::arg("map2") = 2,
             py::arg("map3") = 3)
        .def("setMap0", &Shuffleplanes::setMap0)
        .def("getMap0", &Shuffleplanes::getMap0)
        .def("setMap1", &Shuffleplanes::setMap1)
        .def("getMap1", &Shuffleplanes::getMap1)
        .def("setMap2", &Shuffleplanes::setMap2)
        .def("getMap2", &Shuffleplanes::getMap2)
        .def("setMap3", &Shuffleplanes::setMap3)
        .def("getMap3", &Shuffleplanes::getMap3)
        ;
}