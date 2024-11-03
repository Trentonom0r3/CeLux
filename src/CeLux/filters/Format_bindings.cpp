#include "Format_bindings.hpp"

namespace py = pybind11;

void bind_Format(py::module_ &m) {
    py::class_<Format, FilterBase, std::shared_ptr<Format>>(m, "Format")
        .def(py::init<std::vector<std::string>, std::string, std::string>(),
             py::arg("pix_fmts") = std::vector<std::string>(),
             py::arg("color_spaces") = "",
             py::arg("color_ranges") = "")
        .def("setPix_fmts", &Format::setPix_fmts)
        .def("getPix_fmts", &Format::getPix_fmts)
        .def("setColor_spaces", &Format::setColor_spaces)
        .def("getColor_spaces", &Format::getColor_spaces)
        .def("setColor_ranges", &Format::setColor_ranges)
        .def("getColor_ranges", &Format::getColor_ranges)
        ;
}