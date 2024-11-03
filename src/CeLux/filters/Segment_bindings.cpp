#include "Segment_bindings.hpp"

namespace py = pybind11;

void bind_Segment(py::module_ &m) {
    py::class_<Segment, FilterBase, std::shared_ptr<Segment>>(m, "Segment")
        .def(py::init<std::string, std::string>(),
             py::arg("timestamps") = "",
             py::arg("frames") = "")
        .def("setTimestamps", &Segment::setTimestamps)
        .def("getTimestamps", &Segment::getTimestamps)
        .def("setFrames", &Segment::setFrames)
        .def("getFrames", &Segment::getFrames)
        ;
}