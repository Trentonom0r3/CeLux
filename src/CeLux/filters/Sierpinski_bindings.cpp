#include "Sierpinski_bindings.hpp"

namespace py = pybind11;

void bind_Sierpinski(py::module_ &m) {
    py::class_<Sierpinski, FilterBase, std::shared_ptr<Sierpinski>>(m, "Sierpinski")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, int, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("seed") = 0,
             py::arg("jump") = 100,
             py::arg("type") = 0)
        .def("setSize", &Sierpinski::setSize)
        .def("getSize", &Sierpinski::getSize)
        .def("setRate", &Sierpinski::setRate)
        .def("getRate", &Sierpinski::getRate)
        .def("setSeed", &Sierpinski::setSeed)
        .def("getSeed", &Sierpinski::getSeed)
        .def("setJump", &Sierpinski::setJump)
        .def("getJump", &Sierpinski::getJump)
        .def("setType", &Sierpinski::setType)
        .def("getType", &Sierpinski::getType)
        ;
}