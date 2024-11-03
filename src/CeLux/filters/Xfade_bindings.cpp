#include "Xfade_bindings.hpp"

namespace py = pybind11;

void bind_Xfade(py::module_ &m) {
    py::class_<Xfade, FilterBase, std::shared_ptr<Xfade>>(m, "Xfade")
        .def(py::init<int, int64_t, int64_t, std::string>(),
             py::arg("transition") = 0,
             py::arg("duration") = 1000000ULL,
             py::arg("offset") = 0ULL,
             py::arg("expr") = "")
        .def("setTransition", &Xfade::setTransition)
        .def("getTransition", &Xfade::getTransition)
        .def("setDuration", &Xfade::setDuration)
        .def("getDuration", &Xfade::getDuration)
        .def("setOffset", &Xfade::setOffset)
        .def("getOffset", &Xfade::getOffset)
        .def("setExpr", &Xfade::setExpr)
        .def("getExpr", &Xfade::getExpr)
        ;
}