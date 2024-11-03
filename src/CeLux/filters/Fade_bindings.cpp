#include "Fade_bindings.hpp"

namespace py = pybind11;

void bind_Fade(py::module_ &m) {
    py::class_<Fade, FilterBase, std::shared_ptr<Fade>>(m, "Fade")
        .def(py::init<int, int, int, bool, int64_t, int64_t, std::string>(),
             py::arg("type") = 0,
             py::arg("start_frame") = 0,
             py::arg("nb_frames") = 25,
             py::arg("alpha") = false,
             py::arg("start_time") = 0ULL,
             py::arg("duration") = 0ULL,
             py::arg("color") = "black")
        .def("setType", &Fade::setType)
        .def("getType", &Fade::getType)
        .def("setStart_frame", &Fade::setStart_frame)
        .def("getStart_frame", &Fade::getStart_frame)
        .def("setNb_frames", &Fade::setNb_frames)
        .def("getNb_frames", &Fade::getNb_frames)
        .def("setAlpha", &Fade::setAlpha)
        .def("getAlpha", &Fade::getAlpha)
        .def("setStart_time", &Fade::setStart_time)
        .def("getStart_time", &Fade::getStart_time)
        .def("setDuration", &Fade::setDuration)
        .def("getDuration", &Fade::getDuration)
        .def("setColor", &Fade::setColor)
        .def("getColor", &Fade::getColor)
        ;
}