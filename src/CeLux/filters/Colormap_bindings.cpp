#include "Colormap_bindings.hpp"

namespace py = pybind11;

void bind_Colormap(py::module_ &m) {
    py::class_<Colormap, FilterBase, std::shared_ptr<Colormap>>(m, "Colormap")
        .def(py::init<std::pair<int, int>, int, int, int>(),
             py::arg("patch_size") = std::make_pair<int, int>(0, 1),
             py::arg("nb_patches") = 0,
             py::arg("type") = 1,
             py::arg("kernel") = 0)
        .def("setPatch_size", &Colormap::setPatch_size)
        .def("getPatch_size", &Colormap::getPatch_size)
        .def("setNb_patches", &Colormap::setNb_patches)
        .def("getNb_patches", &Colormap::getNb_patches)
        .def("setType", &Colormap::setType)
        .def("getType", &Colormap::getType)
        .def("setKernel", &Colormap::setKernel)
        .def("getKernel", &Colormap::getKernel)
        ;
}