#include "Crop_bindings.hpp"

namespace py = pybind11;

void bind_Crop(py::module_ &m) {
    py::class_<Crop, FilterBase, std::shared_ptr<Crop>>(m, "Crop")
        .def(py::init<std::string, std::string, std::string, std::string, bool, bool>(),
             py::arg("out_w") = "iw",
             py::arg("out_h") = "ih",
             py::arg("xCropArea") = "(in_w-out_w)/2",
             py::arg("yCropArea") = "(in_h-out_h)/2",
             py::arg("keep_aspect") = false,
             py::arg("exact") = false)
        .def("setOut_w", &Crop::setOut_w)
        .def("getOut_w", &Crop::getOut_w)
        .def("setOut_h", &Crop::setOut_h)
        .def("getOut_h", &Crop::getOut_h)
        .def("setXCropArea", &Crop::setXCropArea)
        .def("getXCropArea", &Crop::getXCropArea)
        .def("setYCropArea", &Crop::setYCropArea)
        .def("getYCropArea", &Crop::getYCropArea)
        .def("setKeep_aspect", &Crop::setKeep_aspect)
        .def("getKeep_aspect", &Crop::getKeep_aspect)
        .def("setExact", &Crop::setExact)
        .def("getExact", &Crop::getExact)
        ;
}