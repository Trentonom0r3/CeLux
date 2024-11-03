#include "Tblend_bindings.hpp"

namespace py = pybind11;

void bind_Tblend(py::module_ &m) {
    py::class_<Tblend, FilterBase, std::shared_ptr<Tblend>>(m, "Tblend")
        .def(py::init<int, int, int, int, int, std::string, std::string, std::string, std::string, std::string, double, double, double, double, double>(),
             py::arg("c0_mode") = 0,
             py::arg("c1_mode") = 0,
             py::arg("c2_mode") = 0,
             py::arg("c3_mode") = 0,
             py::arg("all_mode") = -1,
             py::arg("c0_expr") = "",
             py::arg("c1_expr") = "",
             py::arg("c2_expr") = "",
             py::arg("c3_expr") = "",
             py::arg("all_expr") = "",
             py::arg("c0_opacity") = 1.00,
             py::arg("c1_opacity") = 1.00,
             py::arg("c2_opacity") = 1.00,
             py::arg("c3_opacity") = 1.00,
             py::arg("all_opacity") = 1.00)
        .def("setC0_mode", &Tblend::setC0_mode)
        .def("getC0_mode", &Tblend::getC0_mode)
        .def("setC1_mode", &Tblend::setC1_mode)
        .def("getC1_mode", &Tblend::getC1_mode)
        .def("setC2_mode", &Tblend::setC2_mode)
        .def("getC2_mode", &Tblend::getC2_mode)
        .def("setC3_mode", &Tblend::setC3_mode)
        .def("getC3_mode", &Tblend::getC3_mode)
        .def("setAll_mode", &Tblend::setAll_mode)
        .def("getAll_mode", &Tblend::getAll_mode)
        .def("setC0_expr", &Tblend::setC0_expr)
        .def("getC0_expr", &Tblend::getC0_expr)
        .def("setC1_expr", &Tblend::setC1_expr)
        .def("getC1_expr", &Tblend::getC1_expr)
        .def("setC2_expr", &Tblend::setC2_expr)
        .def("getC2_expr", &Tblend::getC2_expr)
        .def("setC3_expr", &Tblend::setC3_expr)
        .def("getC3_expr", &Tblend::getC3_expr)
        .def("setAll_expr", &Tblend::setAll_expr)
        .def("getAll_expr", &Tblend::getAll_expr)
        .def("setC0_opacity", &Tblend::setC0_opacity)
        .def("getC0_opacity", &Tblend::getC0_opacity)
        .def("setC1_opacity", &Tblend::setC1_opacity)
        .def("getC1_opacity", &Tblend::getC1_opacity)
        .def("setC2_opacity", &Tblend::setC2_opacity)
        .def("getC2_opacity", &Tblend::getC2_opacity)
        .def("setC3_opacity", &Tblend::setC3_opacity)
        .def("getC3_opacity", &Tblend::getC3_opacity)
        .def("setAll_opacity", &Tblend::setAll_opacity)
        .def("getAll_opacity", &Tblend::getAll_opacity)
        ;
}