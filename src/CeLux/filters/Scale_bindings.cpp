#include "Scale_bindings.hpp"

namespace py = pybind11;

void bind_Scale(py::module_ &m) {
    py::class_<Scale, FilterBase, std::shared_ptr<Scale>>(m, "Scale")
        .def(py::init<std::string, std::string, std::string, bool, std::string, int, int, int, int, int, int, int, int, int, int, double, double, int>(),
             py::arg("width") = "",
             py::arg("height") = "",
             py::arg("flags") = "",
             py::arg("interl") = false,
             py::arg("size") = "",
             py::arg("in_color_matrix") = -1,
             py::arg("out_color_matrix") = 2,
             py::arg("in_range") = 0,
             py::arg("out_range") = 0,
             py::arg("in_v_chr_pos") = -513,
             py::arg("in_h_chr_pos") = -513,
             py::arg("out_v_chr_pos") = -513,
             py::arg("out_h_chr_pos") = -513,
             py::arg("force_original_aspect_ratio") = 0,
             py::arg("force_divisible_by") = 1,
             py::arg("param0") = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00,
             py::arg("param1") = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00,
             py::arg("eval") = 0)
        .def("setWidth", &Scale::setWidth)
        .def("getWidth", &Scale::getWidth)
        .def("setHeight", &Scale::setHeight)
        .def("getHeight", &Scale::getHeight)
        .def("setFlags", &Scale::setFlags)
        .def("getFlags", &Scale::getFlags)
        .def("setInterl", &Scale::setInterl)
        .def("getInterl", &Scale::getInterl)
        .def("setSize", &Scale::setSize)
        .def("getSize", &Scale::getSize)
        .def("setIn_color_matrix", &Scale::setIn_color_matrix)
        .def("getIn_color_matrix", &Scale::getIn_color_matrix)
        .def("setOut_color_matrix", &Scale::setOut_color_matrix)
        .def("getOut_color_matrix", &Scale::getOut_color_matrix)
        .def("setIn_range", &Scale::setIn_range)
        .def("getIn_range", &Scale::getIn_range)
        .def("setOut_range", &Scale::setOut_range)
        .def("getOut_range", &Scale::getOut_range)
        .def("setIn_v_chr_pos", &Scale::setIn_v_chr_pos)
        .def("getIn_v_chr_pos", &Scale::getIn_v_chr_pos)
        .def("setIn_h_chr_pos", &Scale::setIn_h_chr_pos)
        .def("getIn_h_chr_pos", &Scale::getIn_h_chr_pos)
        .def("setOut_v_chr_pos", &Scale::setOut_v_chr_pos)
        .def("getOut_v_chr_pos", &Scale::getOut_v_chr_pos)
        .def("setOut_h_chr_pos", &Scale::setOut_h_chr_pos)
        .def("getOut_h_chr_pos", &Scale::getOut_h_chr_pos)
        .def("setForce_original_aspect_ratio", &Scale::setForce_original_aspect_ratio)
        .def("getForce_original_aspect_ratio", &Scale::getForce_original_aspect_ratio)
        .def("setForce_divisible_by", &Scale::setForce_divisible_by)
        .def("getForce_divisible_by", &Scale::getForce_divisible_by)
        .def("setParam0", &Scale::setParam0)
        .def("getParam0", &Scale::getParam0)
        .def("setParam1", &Scale::setParam1)
        .def("getParam1", &Scale::getParam1)
        .def("setEval", &Scale::setEval)
        .def("getEval", &Scale::getEval)
        ;
}