#include "Latency_bindings.hpp"

namespace py = pybind11;

void bind_Latency(py::module_ &m) {
    py::class_<Latency, FilterBase, std::shared_ptr<Latency>>(m, "Latency")
        .def(py::init<>())
        ;
}