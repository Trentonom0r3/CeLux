#include "Showinfo_bindings.hpp"

namespace py = pybind11;

void bind_Showinfo(py::module_ &m) {
    py::class_<Showinfo, FilterBase, std::shared_ptr<Showinfo>>(m, "Showinfo")
        .def(py::init<bool, bool>(),
             py::arg("checksum") = true,
             py::arg("udu_sei_as_ascii") = false)
        .def("setChecksum", &Showinfo::setChecksum)
        .def("getChecksum", &Showinfo::getChecksum)
        .def("setUdu_sei_as_ascii", &Showinfo::setUdu_sei_as_ascii)
        .def("getUdu_sei_as_ascii", &Showinfo::getUdu_sei_as_ascii)
        ;
}