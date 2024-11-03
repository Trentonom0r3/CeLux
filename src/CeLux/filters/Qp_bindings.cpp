#include "Qp_bindings.hpp"

namespace py = pybind11;

void bind_Qp(py::module_ &m) {
    py::class_<Qp, FilterBase, std::shared_ptr<Qp>>(m, "Qp")
        .def(py::init<std::string>(),
             py::arg("qp") = "")
        .def("setQp", &Qp::setQp)
        .def("getQp", &Qp::getQp)
        ;
}