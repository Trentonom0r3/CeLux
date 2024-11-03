#include "Vmafmotion_bindings.hpp"

namespace py = pybind11;

void bind_Vmafmotion(py::module_ &m) {
    py::class_<Vmafmotion, FilterBase, std::shared_ptr<Vmafmotion>>(m, "Vmafmotion")
        .def(py::init<std::string>(),
             py::arg("stats_file") = "")
        .def("setStats_file", &Vmafmotion::setStats_file)
        .def("getStats_file", &Vmafmotion::getStats_file)
        ;
}