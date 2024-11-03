#include "Buffersink_bindings.hpp"

namespace py = pybind11;

void bind_Buffersink(py::module_ &m) {
    py::class_<Buffersink, FilterBase, std::shared_ptr<Buffersink>>(m, "Buffersink")
        .def(py::init<std::vector<std::string>, std::vector<uint8_t>, std::vector<uint8_t>>(),
             py::arg("pix_fmts") = std::vector<std::string>(),
             py::arg("color_spaces") = std::vector<uint8_t>(),
             py::arg("color_ranges") = std::vector<uint8_t>())
        .def("setPix_fmts", &Buffersink::setPix_fmts)
        .def("getPix_fmts", &Buffersink::getPix_fmts)
        .def("setColor_spaces", &Buffersink::setColor_spaces)
        .def("getColor_spaces", &Buffersink::getColor_spaces)
        .def("setColor_ranges", &Buffersink::setColor_ranges)
        .def("getColor_ranges", &Buffersink::getColor_ranges)
        ;
}