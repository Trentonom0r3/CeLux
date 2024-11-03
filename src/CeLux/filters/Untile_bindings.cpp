#include "Untile_bindings.hpp"

namespace py = pybind11;

void bind_Untile(py::module_ &m) {
    py::class_<Untile, FilterBase, std::shared_ptr<Untile>>(m, "Untile")
        .def(py::init<std::pair<int, int>>(),
             py::arg("layout") = std::make_pair<int, int>(0, 1))
        .def("setLayout", &Untile::setLayout)
        .def("getLayout", &Untile::getLayout)
        ;
}