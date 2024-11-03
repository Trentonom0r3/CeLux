#include "Ddagrab_bindings.hpp"

namespace py = pybind11;

void bind_Ddagrab(py::module_ &m) {
    py::class_<Ddagrab, FilterBase, std::shared_ptr<Ddagrab>>(m, "Ddagrab")
        .def(py::init<int, bool, std::pair<int, int>, std::pair<int, int>, int, int, int, bool, bool, bool>(),
             py::arg("output_idx") = 0,
             py::arg("draw_mouse") = true,
             py::arg("framerate") = std::make_pair<int, int>(0, 1),
             py::arg("video_size") = std::make_pair<int, int>(0, 1),
             py::arg("offset_x") = 0,
             py::arg("offset_y") = 0,
             py::arg("output_fmt") = 87,
             py::arg("allow_fallback") = false,
             py::arg("force_fmt") = false,
             py::arg("dup_frames") = true)
        .def("setOutput_idx", &Ddagrab::setOutput_idx)
        .def("getOutput_idx", &Ddagrab::getOutput_idx)
        .def("setDraw_mouse", &Ddagrab::setDraw_mouse)
        .def("getDraw_mouse", &Ddagrab::getDraw_mouse)
        .def("setFramerate", &Ddagrab::setFramerate)
        .def("getFramerate", &Ddagrab::getFramerate)
        .def("setVideo_size", &Ddagrab::setVideo_size)
        .def("getVideo_size", &Ddagrab::getVideo_size)
        .def("setOffset_x", &Ddagrab::setOffset_x)
        .def("getOffset_x", &Ddagrab::getOffset_x)
        .def("setOffset_y", &Ddagrab::setOffset_y)
        .def("getOffset_y", &Ddagrab::getOffset_y)
        .def("setOutput_fmt", &Ddagrab::setOutput_fmt)
        .def("getOutput_fmt", &Ddagrab::getOutput_fmt)
        .def("setAllow_fallback", &Ddagrab::setAllow_fallback)
        .def("getAllow_fallback", &Ddagrab::getAllow_fallback)
        .def("setForce_fmt", &Ddagrab::setForce_fmt)
        .def("getForce_fmt", &Ddagrab::getForce_fmt)
        .def("setDup_frames", &Ddagrab::setDup_frames)
        .def("getDup_frames", &Ddagrab::getDup_frames)
        ;
}