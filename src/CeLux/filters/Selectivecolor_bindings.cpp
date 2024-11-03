#include "Selectivecolor_bindings.hpp"

namespace py = pybind11;

void bind_Selectivecolor(py::module_ &m) {
    py::class_<Selectivecolor, FilterBase, std::shared_ptr<Selectivecolor>>(m, "Selectivecolor")
        .def(py::init<int, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string>(),
             py::arg("correction_method") = 0,
             py::arg("reds") = "",
             py::arg("yellows") = "",
             py::arg("greens") = "",
             py::arg("cyans") = "",
             py::arg("blues") = "",
             py::arg("magentas") = "",
             py::arg("whites") = "",
             py::arg("neutrals") = "",
             py::arg("blacks") = "",
             py::arg("psfile") = "")
        .def("setCorrection_method", &Selectivecolor::setCorrection_method)
        .def("getCorrection_method", &Selectivecolor::getCorrection_method)
        .def("setReds", &Selectivecolor::setReds)
        .def("getReds", &Selectivecolor::getReds)
        .def("setYellows", &Selectivecolor::setYellows)
        .def("getYellows", &Selectivecolor::getYellows)
        .def("setGreens", &Selectivecolor::setGreens)
        .def("getGreens", &Selectivecolor::getGreens)
        .def("setCyans", &Selectivecolor::setCyans)
        .def("getCyans", &Selectivecolor::getCyans)
        .def("setBlues", &Selectivecolor::setBlues)
        .def("getBlues", &Selectivecolor::getBlues)
        .def("setMagentas", &Selectivecolor::setMagentas)
        .def("getMagentas", &Selectivecolor::getMagentas)
        .def("setWhites", &Selectivecolor::setWhites)
        .def("getWhites", &Selectivecolor::getWhites)
        .def("setNeutrals", &Selectivecolor::setNeutrals)
        .def("getNeutrals", &Selectivecolor::getNeutrals)
        .def("setBlacks", &Selectivecolor::setBlacks)
        .def("getBlacks", &Selectivecolor::getBlacks)
        .def("setPsfile", &Selectivecolor::setPsfile)
        .def("getPsfile", &Selectivecolor::getPsfile)
        ;
}