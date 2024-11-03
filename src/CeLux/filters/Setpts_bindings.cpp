#include "Setpts_bindings.hpp"

namespace py = pybind11;

void bind_Setpts(py::module_ &m) {
    py::class_<Setpts, FilterBase, std::shared_ptr<Setpts>>(m, "Setpts")
        .def(py::init<std::string>(),
             py::arg("expr") = "PTS")
        .def("setExpr", &Setpts::setExpr)
        .def("getExpr", &Setpts::getExpr)
        ;
}