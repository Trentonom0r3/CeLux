#include "Blend_bindings.hpp"

namespace py = pybind11;

void bind_Blend(py::module_ &m) {
    py::class_<Blend, FilterBase, std::shared_ptr<Blend>>(m, "Blend")
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
        .def("setC0_mode", &Blend::setC0_mode)
        .def("getC0_mode", &Blend::getC0_mode)
        .def("setC1_mode", &Blend::setC1_mode)
        .def("getC1_mode", &Blend::getC1_mode)
        .def("setC2_mode", &Blend::setC2_mode)
        .def("getC2_mode", &Blend::getC2_mode)
        .def("setC3_mode", &Blend::setC3_mode)
        .def("getC3_mode", &Blend::getC3_mode)
        .def("setAll_mode", &Blend::setAll_mode)
        .def("getAll_mode", &Blend::getAll_mode)
        .def("setC0_expr", &Blend::setC0_expr)
        .def("getC0_expr", &Blend::getC0_expr)
        .def("setC1_expr", &Blend::setC1_expr)
        .def("getC1_expr", &Blend::getC1_expr)
        .def("setC2_expr", &Blend::setC2_expr)
        .def("getC2_expr", &Blend::getC2_expr)
        .def("setC3_expr", &Blend::setC3_expr)
        .def("getC3_expr", &Blend::getC3_expr)
        .def("setAll_expr", &Blend::setAll_expr)
        .def("getAll_expr", &Blend::getAll_expr)
        .def("setC0_opacity", &Blend::setC0_opacity)
        .def("getC0_opacity", &Blend::getC0_opacity)
        .def("setC1_opacity", &Blend::setC1_opacity)
        .def("getC1_opacity", &Blend::getC1_opacity)
        .def("setC2_opacity", &Blend::setC2_opacity)
        .def("getC2_opacity", &Blend::getC2_opacity)
        .def("setC3_opacity", &Blend::setC3_opacity)
        .def("getC3_opacity", &Blend::getC3_opacity)
        .def("setAll_opacity", &Blend::setAll_opacity)
        .def("getAll_opacity", &Blend::getAll_opacity)
        ;
}