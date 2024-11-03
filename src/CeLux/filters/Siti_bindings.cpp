#include "Siti_bindings.hpp"

namespace py = pybind11;

void bind_Siti(py::module_ &m) {
    py::class_<Siti, FilterBase, std::shared_ptr<Siti>>(m, "Siti")
        .def(py::init<bool>(),
             py::arg("print_summary") = false)
        .def("setPrint_summary", &Siti::setPrint_summary)
        .def("getPrint_summary", &Siti::getPrint_summary)
        ;
}