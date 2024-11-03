#include "Dejudder_bindings.hpp"

namespace py = pybind11;

void bind_Dejudder(py::module_ &m) {
    py::class_<Dejudder, FilterBase, std::shared_ptr<Dejudder>>(m, "Dejudder")
        .def(py::init<int>(),
             py::arg("cycle") = 4)
        .def("setCycle", &Dejudder::setCycle)
        .def("getCycle", &Dejudder::getCycle)
        ;
}