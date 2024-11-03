#include "V360_bindings.hpp"

namespace py = pybind11;

void bind_V360(py::module_ &m) {
    py::class_<V360, FilterBase, std::shared_ptr<V360>>(m, "V360")
        .def(py::init<int, int, int, int, int, int, int, std::string, std::string, std::string, std::string, float, float, int, int, float, float, float, std::string, float, float, float, bool, bool, bool, bool, bool, bool, bool, float, float, float, float, float, bool, bool>(),
             py::arg("input") = 0,
             py::arg("output") = 1,
             py::arg("interp") = 1,
             py::arg("outputWidth") = 0,
             py::arg("outputHeight") = 0,
             py::arg("in_stereo") = 0,
             py::arg("out_stereo") = 0,
             py::arg("in_forder") = "rludfb",
             py::arg("out_forder") = "rludfb",
             py::arg("in_frot") = "000000",
             py::arg("out_frot") = "000000",
             py::arg("in_pad") = 0.00,
             py::arg("out_pad") = 0.00,
             py::arg("fin_pad") = 0,
             py::arg("fout_pad") = 0,
             py::arg("yaw") = 0.00,
             py::arg("pitch") = 0.00,
             py::arg("roll") = 0.00,
             py::arg("rorder") = "ypr",
             py::arg("h_fov") = 0.00,
             py::arg("v_fov") = 0.00,
             py::arg("d_fov") = 0.00,
             py::arg("h_flip") = false,
             py::arg("v_flip") = false,
             py::arg("d_flip") = false,
             py::arg("ih_flip") = false,
             py::arg("iv_flip") = false,
             py::arg("in_trans") = false,
             py::arg("out_trans") = false,
             py::arg("ih_fov") = 0.00,
             py::arg("iv_fov") = 0.00,
             py::arg("id_fov") = 0.00,
             py::arg("h_offset") = 0.00,
             py::arg("v_offset") = 0.00,
             py::arg("alpha_mask") = false,
             py::arg("reset_rot") = false)
        .def("setInput", &V360::setInput)
        .def("getInput", &V360::getInput)
        .def("setOutput", &V360::setOutput)
        .def("getOutput", &V360::getOutput)
        .def("setInterp", &V360::setInterp)
        .def("getInterp", &V360::getInterp)
        .def("setOutputWidth", &V360::setOutputWidth)
        .def("getOutputWidth", &V360::getOutputWidth)
        .def("setOutputHeight", &V360::setOutputHeight)
        .def("getOutputHeight", &V360::getOutputHeight)
        .def("setIn_stereo", &V360::setIn_stereo)
        .def("getIn_stereo", &V360::getIn_stereo)
        .def("setOut_stereo", &V360::setOut_stereo)
        .def("getOut_stereo", &V360::getOut_stereo)
        .def("setIn_forder", &V360::setIn_forder)
        .def("getIn_forder", &V360::getIn_forder)
        .def("setOut_forder", &V360::setOut_forder)
        .def("getOut_forder", &V360::getOut_forder)
        .def("setIn_frot", &V360::setIn_frot)
        .def("getIn_frot", &V360::getIn_frot)
        .def("setOut_frot", &V360::setOut_frot)
        .def("getOut_frot", &V360::getOut_frot)
        .def("setIn_pad", &V360::setIn_pad)
        .def("getIn_pad", &V360::getIn_pad)
        .def("setOut_pad", &V360::setOut_pad)
        .def("getOut_pad", &V360::getOut_pad)
        .def("setFin_pad", &V360::setFin_pad)
        .def("getFin_pad", &V360::getFin_pad)
        .def("setFout_pad", &V360::setFout_pad)
        .def("getFout_pad", &V360::getFout_pad)
        .def("setYaw", &V360::setYaw)
        .def("getYaw", &V360::getYaw)
        .def("setPitch", &V360::setPitch)
        .def("getPitch", &V360::getPitch)
        .def("setRoll", &V360::setRoll)
        .def("getRoll", &V360::getRoll)
        .def("setRorder", &V360::setRorder)
        .def("getRorder", &V360::getRorder)
        .def("setH_fov", &V360::setH_fov)
        .def("getH_fov", &V360::getH_fov)
        .def("setV_fov", &V360::setV_fov)
        .def("getV_fov", &V360::getV_fov)
        .def("setD_fov", &V360::setD_fov)
        .def("getD_fov", &V360::getD_fov)
        .def("setH_flip", &V360::setH_flip)
        .def("getH_flip", &V360::getH_flip)
        .def("setV_flip", &V360::setV_flip)
        .def("getV_flip", &V360::getV_flip)
        .def("setD_flip", &V360::setD_flip)
        .def("getD_flip", &V360::getD_flip)
        .def("setIh_flip", &V360::setIh_flip)
        .def("getIh_flip", &V360::getIh_flip)
        .def("setIv_flip", &V360::setIv_flip)
        .def("getIv_flip", &V360::getIv_flip)
        .def("setIn_trans", &V360::setIn_trans)
        .def("getIn_trans", &V360::getIn_trans)
        .def("setOut_trans", &V360::setOut_trans)
        .def("getOut_trans", &V360::getOut_trans)
        .def("setIh_fov", &V360::setIh_fov)
        .def("getIh_fov", &V360::getIh_fov)
        .def("setIv_fov", &V360::setIv_fov)
        .def("getIv_fov", &V360::getIv_fov)
        .def("setId_fov", &V360::setId_fov)
        .def("getId_fov", &V360::getId_fov)
        .def("setH_offset", &V360::setH_offset)
        .def("getH_offset", &V360::getH_offset)
        .def("setV_offset", &V360::setV_offset)
        .def("getV_offset", &V360::getV_offset)
        .def("setAlpha_mask", &V360::setAlpha_mask)
        .def("getAlpha_mask", &V360::getAlpha_mask)
        .def("setReset_rot", &V360::setReset_rot)
        .def("getReset_rot", &V360::getReset_rot)
        ;
}