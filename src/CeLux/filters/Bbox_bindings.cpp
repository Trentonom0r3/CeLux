#include "Bbox_bindings.hpp"

namespace py = pybind11;

void bind_Bbox(py::module_ &m) {
    py::class_<Bbox, FilterBase, std::shared_ptr<Bbox>>(m, "Bbox")
        .def(py::init<int>(),
             py::arg("min_val") = 16)
        .def("setMin_val", &Bbox::setMin_val)
        .def("getMin_val", &Bbox::getMin_val)
        ;
}