#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "c_rank_nicolson.h"
#include "2d_c_rank_nicolson.h"

using namespace Eigen;

// PYTHON BINDING!

PYBIND11_MODULE(c_rank_nicolson, m)
{
    m.doc() = "Python wrapping of a c++ Crank-Nicolson integrator";
    py::class_<crank_nicolson>(m, "crank_nicolson")
        .def(py::init<size_t, double, double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>())
        .def("N", &crank_nicolson::N, "Get number of samples.")
        .def("XMIN", &crank_nicolson::XMIN, "Get minimum of interval.")
        .def("XMAX", &crank_nicolson::XMAX, "Get maximum of interval.")
        .def("DT", &crank_nicolson::DT, "Get dt.")
        .def("DX", &crank_nicolson::DX, "Get dx.")
        .def("A", &crank_nicolson::A, "Get A vector.")
        .def("B", &crank_nicolson::B, "Get B vector.")
        .def("C", &crank_nicolson::C, "Get C vector.")
        .def("C", &crank_nicolson::D, "Get D vector.")
        .def("a_left", &crank_nicolson::a_left)
        .def("b_left", &crank_nicolson::b_left)
        .def("c_left", &crank_nicolson::c_left)
        .def("a_right", &crank_nicolson::a_right)
        .def("b_right", &crank_nicolson::b_right)
        .def("c_right", &crank_nicolson::c_right)
        .def("f", &crank_nicolson::f)
        .def("c_prime", &crank_nicolson::c_prime)
        .def("f_prime", &crank_nicolson::f_prime)
        .def("x", &crank_nicolson::x, "Get rho(x) data.")
        .def("x0", &crank_nicolson::x0, "Get rho_0(x) data.")
        .def("set_executed_iterations", &crank_nicolson::set_executed_iterations, "Set number of executed iterations.")
        .def("set_x_values", &crank_nicolson::set_x_values, "Set new density vector.")
        .def("set_source", &crank_nicolson::set_source, "Set new source values")
        .def("remove_source", &crank_nicolson::remove_source, "Remove source values")
        .def("lock_left", &crank_nicolson::lock_left, "Lock left border")
        .def("lock_right", &crank_nicolson::lock_right, "Lock right border")
        .def("unlock_left", &crank_nicolson::unlock_left, "Unlock left border")
        .def("unlock_right", &crank_nicolson::unlock_right, "Unlock right border")
        .def("executed_iterations", &crank_nicolson::executed_iterations, "Get number of executed iterations.")
        .def("iterate", &crank_nicolson::iterate, "Perform requested iterations.")
        .def("reset", &crank_nicolson::reset, "Reset the integrator.");

    py::class_<crank_nicolson_2d>(m, "crank_nicolson_2d")
        .def(py::init<size_t, size_t, double, double, double, double, double, MatrixXdr, MatrixXdr, MatrixXdr, MatrixXdr, MatrixXdr, MatrixXdr>())
        .def("iterate", &crank_nicolson_2d::iterate, "Iterate simulation.")

        .def("NX", &crank_nicolson_2d::NX, "Get NX")
        .def("NY", &crank_nicolson_2d::NY, "Get NY")
        .def("XMIN", &crank_nicolson_2d::XMIN, "Get XMIN")
        .def("XMIN", &crank_nicolson_2d::XMAX, "Get XMAX")
        .def("YMIN", &crank_nicolson_2d::YMIN, "Get YMIN")
        .def("YMAX", &crank_nicolson_2d::YMAX, "Get YMAX")
        .def("DT", &crank_nicolson_2d::DT, "Get DT")
        .def("DX", &crank_nicolson_2d::DX, "Get DX")
        .def("DY", &crank_nicolson_2d::DY, "Get DY")

        .def("U0_mat", &crank_nicolson_2d::U0_mat, "Get starting matrix")
        .def("U_mat", &crank_nicolson_2d::U_mat, "Get resulting matrix")
        .def("C_x", &crank_nicolson_2d::C_x, "Get C_x")
        .def("C_y", &crank_nicolson_2d::C_y, "Get C_y")
        .def("A_xx", &crank_nicolson_2d::A_xx, "Get A_xx")
        .def("A_yy", &crank_nicolson_2d::A_xx, "Get A_yy")
        .def("A_xy", &crank_nicolson_2d::A_xx, "Get A_xy")

        .def("executed_iterations", &crank_nicolson_2d::executed_iterations, "Get executed iterations");
}