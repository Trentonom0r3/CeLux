#include "Geq_bindings.hpp"

namespace py = pybind11;

void bind_Geq(py::module_ &m) {
    py::class_<Geq, FilterBase, std::shared_ptr<Geq>>(m, "Geq")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string, std::string, int>(),
             py::arg("lum_expr") = "",
             py::arg("cb_expr") = "",
             py::arg("cr_expr") = "",
             py::arg("alpha_expr") = "",
             py::arg("red_expr") = "",
             py::arg("green_expr") = "",
             py::arg("blue_expr") = "",
             py::arg("interpolation") = 1)
        .def("setLum_expr", &Geq::setLum_expr)
        .def("getLum_expr", &Geq::getLum_expr)
        .def("setCb_expr", &Geq::setCb_expr)
        .def("getCb_expr", &Geq::getCb_expr)
        .def("setCr_expr", &Geq::setCr_expr)
        .def("getCr_expr", &Geq::getCr_expr)
        .def("setAlpha_expr", &Geq::setAlpha_expr)
        .def("getAlpha_expr", &Geq::getAlpha_expr)
        .def("setRed_expr", &Geq::setRed_expr)
        .def("getRed_expr", &Geq::getRed_expr)
        .def("setGreen_expr", &Geq::setGreen_expr)
        .def("getGreen_expr", &Geq::getGreen_expr)
        .def("setBlue_expr", &Geq::setBlue_expr)
        .def("getBlue_expr", &Geq::getBlue_expr)
        .def("setInterpolation", &Geq::setInterpolation)
        .def("getInterpolation", &Geq::getInterpolation)
        ;
}