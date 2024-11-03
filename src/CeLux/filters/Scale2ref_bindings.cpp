#include "Scale2ref_bindings.hpp"

namespace py = pybind11;

void bind_Scale2ref(py::module_ &m) {
    py::class_<Scale2ref, FilterBase, std::shared_ptr<Scale2ref>>(m, "Scale2ref")
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
        .def("setWidth", &Scale2ref::setWidth)
        .def("getWidth", &Scale2ref::getWidth)
        .def("setHeight", &Scale2ref::setHeight)
        .def("getHeight", &Scale2ref::getHeight)
        .def("setFlags", &Scale2ref::setFlags)
        .def("getFlags", &Scale2ref::getFlags)
        .def("setInterl", &Scale2ref::setInterl)
        .def("getInterl", &Scale2ref::getInterl)
        .def("setSize", &Scale2ref::setSize)
        .def("getSize", &Scale2ref::getSize)
        .def("setIn_color_matrix", &Scale2ref::setIn_color_matrix)
        .def("getIn_color_matrix", &Scale2ref::getIn_color_matrix)
        .def("setOut_color_matrix", &Scale2ref::setOut_color_matrix)
        .def("getOut_color_matrix", &Scale2ref::getOut_color_matrix)
        .def("setIn_range", &Scale2ref::setIn_range)
        .def("getIn_range", &Scale2ref::getIn_range)
        .def("setOut_range", &Scale2ref::setOut_range)
        .def("getOut_range", &Scale2ref::getOut_range)
        .def("setIn_v_chr_pos", &Scale2ref::setIn_v_chr_pos)
        .def("getIn_v_chr_pos", &Scale2ref::getIn_v_chr_pos)
        .def("setIn_h_chr_pos", &Scale2ref::setIn_h_chr_pos)
        .def("getIn_h_chr_pos", &Scale2ref::getIn_h_chr_pos)
        .def("setOut_v_chr_pos", &Scale2ref::setOut_v_chr_pos)
        .def("getOut_v_chr_pos", &Scale2ref::getOut_v_chr_pos)
        .def("setOut_h_chr_pos", &Scale2ref::setOut_h_chr_pos)
        .def("getOut_h_chr_pos", &Scale2ref::getOut_h_chr_pos)
        .def("setForce_original_aspect_ratio", &Scale2ref::setForce_original_aspect_ratio)
        .def("getForce_original_aspect_ratio", &Scale2ref::getForce_original_aspect_ratio)
        .def("setForce_divisible_by", &Scale2ref::setForce_divisible_by)
        .def("getForce_divisible_by", &Scale2ref::getForce_divisible_by)
        .def("setParam0", &Scale2ref::setParam0)
        .def("getParam0", &Scale2ref::getParam0)
        .def("setParam1", &Scale2ref::setParam1)
        .def("getParam1", &Scale2ref::getParam1)
        .def("setEval", &Scale2ref::setEval)
        .def("getEval", &Scale2ref::getEval)
        ;
}