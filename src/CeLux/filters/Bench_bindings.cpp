#include "Bench_bindings.hpp"

namespace py = pybind11;

void bind_Bench(py::module_ &m) {
    py::class_<Bench, FilterBase, std::shared_ptr<Bench>>(m, "Bench")
        .def(py::init<int>(),
             py::arg("action") = 0)
        .def("setAction", &Bench::setAction)
        .def("getAction", &Bench::getAction)
        ;
}