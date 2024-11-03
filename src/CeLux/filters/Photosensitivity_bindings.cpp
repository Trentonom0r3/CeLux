#include "Photosensitivity_bindings.hpp"

namespace py = pybind11;

void bind_Photosensitivity(py::module_ &m) {
    py::class_<Photosensitivity, FilterBase, std::shared_ptr<Photosensitivity>>(m, "Photosensitivity")
        .def(py::init<int, float, int, bool>(),
             py::arg("frames") = 30,
             py::arg("threshold") = 1.00,
             py::arg("skip") = 1,
             py::arg("bypass") = false)
        .def("setFrames", &Photosensitivity::setFrames)
        .def("getFrames", &Photosensitivity::getFrames)
        .def("setThreshold", &Photosensitivity::setThreshold)
        .def("getThreshold", &Photosensitivity::getThreshold)
        .def("setSkip", &Photosensitivity::setSkip)
        .def("getSkip", &Photosensitivity::getSkip)
        .def("setBypass", &Photosensitivity::setBypass)
        .def("getBypass", &Photosensitivity::getBypass)
        ;
}