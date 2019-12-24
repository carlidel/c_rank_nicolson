//#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <ratio>
#include <chrono>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
typedef Matrix<double, Dynamic, 1> VectorXd;
typedef SparseMatrix<double, RowMajor> SpMat;
typedef Triplet<double> T;

class crank_nicolson_2d
{
private:
    // PARAMETERS
    size_t _NX;
    size_t _NY;
    double _XMIN;
    double _XMAX;
    double _YMIN;
    double _YMAX;
    double _DT;
    double _DX;
    double _DY;

    // COEFFICIENT LIST
    MatrixXdr _U0_mat;
    MatrixXdr _U_mat;
    MatrixXdr _C_x;
    MatrixXdr _C_y;
    MatrixXdr _A_xx;
    MatrixXdr _A_yy;
    MatrixXdr _A_xy;

    // ELEMENTS
    SpMat _W0_rhs;
    SpMat _W0_lhs;
    SpMat _W1_rhs_W0;
    SpMat _W1_rhs_U0;
    SpMat _W1_lhs;
    SpMat _W2_rhs_W1;
    SpMat _W2_rhs_U0;
    SpMat _W2_lhs;
    SpMat _V0_rhs_W0;
    SpMat _V0_rhs_W2;
    SpMat _V0_rhs_U0;
    SpMat _V0_lhs;
    SpMat _V1_rhs_V0;
    SpMat _V1_rhs_U0;
    SpMat _V1_lhs;
    SpMat _V2_rhs_V1;
    SpMat _V2_rhs_U0;
    SpMat _V2_lhs;
    std::vector<T> _W0_rhs_triplet;
    std::vector<T> _W0_lhs_triplet;
    std::vector<T> _W1_rhs_W0_triplet;
    std::vector<T> _W1_rhs_U0_triplet;
    std::vector<T> _W1_lhs_triplet;
    std::vector<T> _W2_rhs_W1_triplet;
    std::vector<T> _W2_rhs_U0_triplet;
    std::vector<T> _W2_lhs_triplet;
    std::vector<T> _V0_rhs_W0_triplet;
    std::vector<T> _V0_rhs_W2_triplet;
    std::vector<T> _V0_rhs_U0_triplet;
    std::vector<T> _V0_lhs_triplet;
    std::vector<T> _V1_rhs_V0_triplet;
    std::vector<T> _V1_rhs_U0_triplet;
    std::vector<T> _V1_lhs_triplet;
    std::vector<T> _V2_rhs_V1_triplet;
    std::vector<T> _V2_rhs_U0_triplet;
    std::vector<T> _V2_lhs_triplet;
    MatrixXdr _U0;
    MatrixXdr _W0;
    MatrixXdr _W1;
    MatrixXdr _W2;
    MatrixXdr _V0;
    MatrixXdr _V1;
    MatrixXdr _V2;
    // Iterative method? (OpenMP!!!)
    BiCGSTAB<SpMat, IncompleteLUT<double>> _W1_solver_it;
    BiCGSTAB<SpMat, IncompleteLUT<double>> _W2_solver_it;
    BiCGSTAB<SpMat, IncompleteLUT<double>> _V1_solver_it;
    BiCGSTAB<SpMat, IncompleteLUT<double>> _V2_solver_it;

    // STATUS
    unsigned int _executed_iterations;

    // UTILIS
    std::chrono::steady_clock::time_point _begin;
    std::chrono::steady_clock::time_point _end;

public:
    // CONSTRUCTORS
    crank_nicolson_2d();
    crank_nicolson_2d(size_t NX, size_t NY, double XMIN, double XMAX, double YMIN, double YMAX, double DT, MatrixXdr U, MatrixXdr C_x = MatrixXdr(), MatrixXdr C_y = MatrixXdr(), MatrixXdr A_xx = MatrixXdr(), MatrixXdr A_yy = MatrixXdr(), MatrixXdr A_xy = MatrixXdr());

    // PUBLIC METHODS
    void iterate(unsigned int n_iterations, bool print_time = false);

    // GETTERS
    const size_t &NX() const;
    const size_t &NY() const;
    const double &XMIN() const;
    const double &XMAX() const;
    const double &YMIN() const;
    const double &YMAX() const;
    const double &DT() const;
    const double &DX() const;
    const double &DY() const;

    const MatrixXdr &U0_mat() const;
    const MatrixXdr &U_mat() const;
    const MatrixXdr &C_x() const;
    const MatrixXdr &C_y() const;
    const MatrixXdr &A_xx() const;
    const MatrixXdr &A_yy() const;
    const MatrixXdr &A_xy() const;

    const unsigned int &executed_iterations() const;

private:
    // PRIVATE METHODS
    size_t convert_coords(const size_t &i, const size_t &j);

    void sampling_x(const size_t &i, const size_t &j, std::vector<T> &target, double scalar = 1.0);
    void sampling_y(const size_t &i, const size_t &j, std::vector<T> &target, double scalar=1.0);
    void sampling_xx(const size_t &i, const size_t &j, std::vector<T> &target, double scalar=1.0);
    void sampling_yy(const size_t &i, const size_t &j, std::vector<T> &target, double scalar=1.0);
    void sampling_xy(const size_t &i, const size_t &j, std::vector<T> &target, double scalar=1.0);

    void I(std::vector<T> &target, double scalar=1.0);
    void F_0(std::vector<T> &target, double scalar=1.0);
    void F_1(std::vector<T> &target, double scalar=1.0);
    void F_2(std::vector<T> &target, double scalar=1.0);
    void F_total(std::vector<T> &target, double scalar = 1.0);

    void build_matrices();
};
