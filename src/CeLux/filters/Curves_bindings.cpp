#include "Curves_bindings.hpp"

namespace py = pybind11;

void bind_Curves(py::module_ &m) {
    py::class_<Curves, FilterBase, std::shared_ptr<Curves>>(m, "Curves")
        .def(py::init<int, std::string, std::string, std::string, std::string, std::string, std::string, std::string, int>(),
             py::arg("preset") = 0,
             py::arg("master") = "",
             py::arg("red") = "",
             py::arg("green") = "",
             py::arg("blue") = "",
             py::arg("all") = "",
             py::arg("psfile") = "",
             py::arg("plot") = "",
             py::arg("interp") = 0)
        .def("setPreset", &Curves::setPreset)
        .def("getPreset", &Curves::getPreset)
        .def("setMaster", &Curves::setMaster)
        .def("getMaster", &Curves::getMaster)
        .def("setRed", &Curves::setRed)
        .def("getRed", &Curves::getRed)
        .def("setGreen", &Curves::setGreen)
        .def("getGreen", &Curves::getGreen)
        .def("setBlue", &Curves::setBlue)
        .def("getBlue", &Curves::getBlue)
        .def("setAll", &Curves::setAll)
        .def("getAll", &Curves::getAll)
        .def("setPsfile", &Curves::setPsfile)
        .def("getPsfile", &Curves::getPsfile)
        .def("setPlot", &Curves::setPlot)
        .def("getPlot", &Curves::getPlot)
        .def("setInterp", &Curves::setInterp)
        .def("getInterp", &Curves::getInterp)
        ;
}