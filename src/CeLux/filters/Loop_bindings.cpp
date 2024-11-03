#include "Loop_bindings.hpp"

namespace py = pybind11;

void bind_Loop(py::module_ &m) {
    py::class_<Loop, FilterBase, std::shared_ptr<Loop>>(m, "Loop")
        .def(py::init<int, int64_t, int64_t, int64_t>(),
             py::arg("loop") = 0,
             py::arg("size") = 0ULL,
             py::arg("start") = 0ULL,
             py::arg("time") = 9223372036854775807ULL)
        .def("setLoop", &Loop::setLoop)
        .def("getLoop", &Loop::getLoop)
        .def("setSize", &Loop::setSize)
        .def("getSize", &Loop::getSize)
        .def("setStart", &Loop::setStart)
        .def("getStart", &Loop::getStart)
        .def("setTime", &Loop::setTime)
        .def("getTime", &Loop::getTime)
        ;
}