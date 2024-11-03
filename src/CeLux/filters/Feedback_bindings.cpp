#include "Feedback_bindings.hpp"

namespace py = pybind11;

void bind_Feedback(py::module_ &m) {
    py::class_<Feedback, FilterBase, std::shared_ptr<Feedback>>(m, "Feedback")
        .def(py::init<int, int>(),
             py::arg("topLeftCropPosition") = 0,
             py::arg("cropSize") = 0)
        .def("setTopLeftCropPosition", &Feedback::setTopLeftCropPosition)
        .def("getTopLeftCropPosition", &Feedback::getTopLeftCropPosition)
        .def("setCropSize", &Feedback::setCropSize)
        .def("getCropSize", &Feedback::getCropSize)
        ;
}