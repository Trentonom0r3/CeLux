#include "Thumbnail_bindings.hpp"

namespace py = pybind11;

void bind_Thumbnail(py::module_ &m) {
    py::class_<Thumbnail, FilterBase, std::shared_ptr<Thumbnail>>(m, "Thumbnail")
        .def(py::init<int, int>(),
             py::arg("framesBatchSize") = 100,
             py::arg("log") = 32)
        .def("setFramesBatchSize", &Thumbnail::setFramesBatchSize)
        .def("getFramesBatchSize", &Thumbnail::getFramesBatchSize)
        .def("setLog", &Thumbnail::setLog)
        .def("getLog", &Thumbnail::getLog)
        ;
}