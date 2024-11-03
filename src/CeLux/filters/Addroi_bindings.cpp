#include "Addroi_bindings.hpp"

namespace py = pybind11;

void bind_Addroi(py::module_ &m) {
    py::class_<Addroi, FilterBase, std::shared_ptr<Addroi>>(m, "Addroi")
        .def(py::init<std::string, std::string, std::string, std::string, std::pair<int, int>, bool>(),
             py::arg("regionDistanceFromLeftEdgeOfFrame") = "0",
             py::arg("regionDistanceFromTopEdgeOfFrame") = "0",
             py::arg("regionWidth") = "0",
             py::arg("regionHeight") = "0",
             py::arg("qoffset") = std::make_pair<int, int>(0, 1),
             py::arg("clear") = false)
        .def("setRegionDistanceFromLeftEdgeOfFrame", &Addroi::setRegionDistanceFromLeftEdgeOfFrame)
        .def("getRegionDistanceFromLeftEdgeOfFrame", &Addroi::getRegionDistanceFromLeftEdgeOfFrame)
        .def("setRegionDistanceFromTopEdgeOfFrame", &Addroi::setRegionDistanceFromTopEdgeOfFrame)
        .def("getRegionDistanceFromTopEdgeOfFrame", &Addroi::getRegionDistanceFromTopEdgeOfFrame)
        .def("setRegionWidth", &Addroi::setRegionWidth)
        .def("getRegionWidth", &Addroi::getRegionWidth)
        .def("setRegionHeight", &Addroi::setRegionHeight)
        .def("getRegionHeight", &Addroi::getRegionHeight)
        .def("setQoffset", &Addroi::setQoffset)
        .def("getQoffset", &Addroi::getQoffset)
        .def("setClear", &Addroi::setClear)
        .def("getClear", &Addroi::getClear)
        ;
}