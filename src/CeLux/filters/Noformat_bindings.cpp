#include "Noformat_bindings.hpp"

namespace py = pybind11;

void bind_Noformat(py::module_ &m) {
    py::class_<Noformat, FilterBase, std::shared_ptr<Noformat>>(m, "Noformat")
        .def(py::init<std::vector<std::string>, std::string, std::string>(),
             py::arg("pix_fmts") = std::vector<std::string>(),
             py::arg("color_spaces") = "",
             py::arg("color_ranges") = "")
        .def("setPix_fmts", &Noformat::setPix_fmts)
        .def("getPix_fmts", &Noformat::getPix_fmts)
        .def("setColor_spaces", &Noformat::setColor_spaces)
        .def("getColor_spaces", &Noformat::getColor_spaces)
        .def("setColor_ranges", &Noformat::setColor_ranges)
        .def("getColor_ranges", &Noformat::getColor_ranges)
        ;
}