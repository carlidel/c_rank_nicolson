#include "2d_c_rank_nicolson.h"

//  crank_nicolson_2d

// CONSTRUCTORS

crank_nicolson_2d::crank_nicolson_2d()
{
}

crank_nicolson_2d::crank_nicolson_2d(size_t NX, size_t NY, double XMIN, double XMAX, double YMIN, double YMAX, double DT, MatrixXdr U, MatrixXdr C_x, MatrixXdr C_y, MatrixXdr A_xx, MatrixXdr A_yy, MatrixXdr A_xy) : _NX(NX), _NY(NY), _XMIN(XMIN), _XMAX(XMAX), _YMIN(YMIN), _YMAX(YMAX), _DT(DT), _U0_mat(U), _U_mat(U), _C_x(C_x), _C_y(C_y), _A_xx(A_xx), _A_yy(A_yy), _A_xy(A_xy)
{
    if (this->_C_x.size() == 0)
        this->_C_x = MatrixXdr::Zero(this->_NX, this->_NY);
    if(this->_C_y.size()==0)
        this->_C_y = MatrixXdr::Zero(this->_NX, this->_NY);
    if (this->_A_xx.size() == 0)
        this->_A_xx = MatrixXdr::Zero(this->_NX, this->_NY);
    if (this->_A_yy.size() == 0)
        this->_A_yy = MatrixXdr::Zero(this->_NX, this->_NY);
    if (this->_A_xy.size() == 0)
        this->_A_xy = MatrixXdr::Zero(this->_NX, this->_NY);
    
    this->_DX = (this->_XMAX - this->_XMIN) / (this->_NX - 1);
    this->_DY = (this->_YMAX - this->_YMIN) / (this->_NY - 1);

    this->_U0 = this->_U_mat;
    this->_U0.resize(this->_NX * this->_NY, 1);
    
    this->build_matrices();
}

// PUBLIC METHODS

void crank_nicolson_2d::iterate(unsigned int n_iterations, bool print_time)
{
    if (print_time)
        this->_begin = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n_iterations; ++i)
    {
        // Increment
        this->_executed_iterations += 1;
        
        // Execute
        this->_W0.noalias() = this->_W0_rhs * this->_U0;

        //this->_W1 = this->_W1_solver.solve(this->_W1_rhs_U0 * this->_U0 + this->_W1_rhs_W0 * this->_W0);
        this->_W1.noalias() = this->_W1_solver_it.solve(this->_W1_rhs_U0 * this->_U0 + this->_W1_rhs_W0 * this->_W0);

        //this->_W2 = this->_W2_solver.solve(this->_W2_rhs_U0 * this->_U0 + this->_W2_rhs_W1 * this->_W1);
        this->_W2.noalias() = this->_W2_solver_it.solve(this->_W2_rhs_U0 * this->_U0 + this->_W2_rhs_W1 * this->_W1);

        this->_V0.noalias() = this->_V0_rhs_U0 * this->_U0 + this->_V0_rhs_W0 * this->_W0 + this->_V0_rhs_W2 * this->_W2;

        //this->_V1 = this->_V1_solver.solve(this->_V1_rhs_U0 * this->_U0 + this->_V1_rhs_V0 * this->_V0);
        this->_V1.noalias() = this->_V1_solver_it.solve(this->_V1_rhs_U0 * this->_U0 + this->_V1_rhs_V0 * this->_V0);

        //this->_V2 = this->_V2_solver.solve(this->_V2_rhs_U0 * this->_U0 + this->_V2_rhs_V1 * this->_V1);
        this->_V2.noalias() = this->_V2_solver_it.solve(this->_V2_rhs_U0 * this->_U0 + this->_V2_rhs_V1 * this->_V1);

        // Conclude
        this->_U0 = this->_V2;
    }
    // Extract
    this->_U_mat = this->_U0;
    this->_U_mat.resize(this->_NX, this->_NY);
    if (print_time)
    {
        this->_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(this->_end - this->_begin);
        std::cout << n_iterations << " iterations done in " << time_span.count() << " seconds.\n"
                  << "Iterations/Seconds = " << n_iterations / time_span.count() << std::endl;
    }
}

// PRIVATE METHODS

size_t crank_nicolson_2d::convert_coords(const size_t &i, const size_t &j)
{
    return j + (this->_NX * i);
}

void crank_nicolson_2d::sampling_x(const size_t &i, const size_t &j, std::vector<T> &target, double scalar)
{
    if (i > 0)
        target.push_back(T(convert_coords(i, j), convert_coords(i - 1, j), -scalar * this->_C_x(i - 1, j) / (2 * this->_DX)));
    if (i < this->_NX - 1)
        target.push_back(T(convert_coords(i, j), convert_coords(i + 1, j), +scalar * this->_C_x(i + 1, j) / (2 * this->_DX)));
}

void crank_nicolson_2d::sampling_y(const size_t &i, const size_t &j, std::vector<T> &target, double scalar)
{
    if (j > 0)
        target.push_back(T(convert_coords(i, j), convert_coords(i, j - 1), -scalar * this->_C_y(i, j - 1) / (2 * this->_DY)));
    if (j < this->_NY - 1)
        target.push_back(T(convert_coords(i, j), convert_coords(i, j + 1), +scalar * this->_C_y(i, j + 1) / (2 * this->_DY)));
}

void crank_nicolson_2d::sampling_xx(const size_t &i, const size_t &j, std::vector<T> &target, double scalar)
{
    if (i > 0)
        target.push_back(T(convert_coords(i, j), convert_coords(i - 1, j), +scalar * this->_A_xx(i - 1, j) / (this->_DX * this->_DX)));
    if (i < this->_NX - 1)
        target.push_back(T(convert_coords(i, j), convert_coords(i + 1, j), +scalar * this->_A_xx(i + 1, j) / (this->_DX * this->_DX)));

    target.push_back(T(convert_coords(i, j), convert_coords(i, j), -2 * scalar * this->_A_xx(i, j) / (this->_DX * this->_DX)));
}

void crank_nicolson_2d::sampling_yy(const size_t &i, const size_t &j, std::vector<T> &target, double scalar)
{
    if (j > 0)
        target.push_back(T(convert_coords(i, j), convert_coords(i, j - 1), +scalar * this->_A_yy(i, j - 1) / (this->_DY * this->_DY)));
    if (j < this->_NY - 1)
        target.push_back(T(convert_coords(i, j), convert_coords(i, j + 1), +scalar * this->_A_yy(i, j + 1) / (this->_DY * this->_DY)));

    target.push_back(T(convert_coords(i, j), convert_coords(i, j), -2 * scalar * this->_A_yy(i, j) / (this->_DY * this->_DY)));
}

void crank_nicolson_2d::sampling_xy(const size_t &i, const size_t &j, std::vector<T> &target, double scalar)
{
    if (i > 0)
    {
        if (j > 0)
            target.push_back(T(convert_coords(i, j), convert_coords(i - 1, j - 1), +scalar * this->_A_xy(i - 1, j - 1) / (4 * this->_DX * this->_DY)));
        if (j < this->_NY - 1)
            target.push_back(T(convert_coords(i, j), convert_coords(i - 1, j + 1), -scalar * this->_A_xy(i - 1, j + 1) / (4 * this->_DX * this->_DY)));
    }
    if (i < this->_NX - 1)
    {
        if (j > 0)
            target.push_back(T(convert_coords(i, j), convert_coords(i + 1, j - 1), -scalar * this->_A_xy(i + 1, j - 1) / (4 * this->_DX * this->_DY)));
        if (j < this->_NY - 1)
            target.push_back(T(convert_coords(i, j), convert_coords(i + 1, j + 1), +scalar * this->_A_xy(i + 1, j + 1) / (4 * this->_DX * this->_DY)));
    }
}

void crank_nicolson_2d::I(std::vector<T> &target, double scalar)
{
    // Add Identity
    for (size_t i = 0; i < this->_NX * this->_NY; ++i)
        target.push_back(T(i, i, scalar));
}

void crank_nicolson_2d::F_0(std::vector<T> &target, double scalar)
{
    // Only mixed terms
    for (size_t i = 0; i < this->_NX; ++i)
        for (size_t j = 0; j < this->_NY; ++j)
            this->sampling_xy(i, j, target, scalar);
}

void crank_nicolson_2d::F_1(std::vector<T> &target, double scalar)
{
    // Only X terms
    for (size_t i = 0; i < this->_NX; ++i)
        for (size_t j = 0; j < this->_NY; ++j)
        {
            this->sampling_x(i, j, target, scalar);
            this->sampling_xx(i, j, target, scalar);
        }
}

void crank_nicolson_2d::F_2(std::vector<T> &target, double scalar)
{
    // Only Y terms
    for (size_t i = 0; i < this->_NX; ++i)
        for (size_t j = 0; j < this->_NY; ++j)
        {
            this->sampling_y(i, j, target, scalar);
            this->sampling_yy(i, j, target, scalar);
        }
}

void crank_nicolson_2d::F_total(std::vector<T> &target, double scalar)
{
    // Everything
    this->F_0(target, scalar);
    this->F_1(target, scalar);
    this->F_2(target, scalar);
}

void crank_nicolson_2d::build_matrices()
{
    // W0 rhs
    this->F_total(_W0_rhs_triplet, this->_DT);
    this->I(this->_W0_rhs_triplet, 1.0);
    // W0 lhs
    this->I(this->_W0_lhs_triplet, 1.0);

    // W1 rhs W0
    this->I(this->_W1_rhs_W0_triplet, 1.0);
    // W1 rhs U0
    this->F_1(this->_W1_rhs_U0_triplet, -0.5 * this->_DT);
    // W1 lhs
    this->F_1(this->_W1_lhs_triplet, -0.5 * this->_DT);
    this->I(this->_W1_lhs_triplet, 1.0);

    // W2 rhs W1
    this->I(this->_W2_rhs_W1_triplet, 1.0);
    // W2 rhs U0
    this->F_2(this->_W2_rhs_U0_triplet, -0.5 * this->_DT);
    // W2 lhs
    this->F_2(this->_W2_lhs_triplet, -0.5 * this->_DT);
    this->I(this->_W2_lhs_triplet, 1.0);

    // V0 rhs W0
    this->I(this->_V0_rhs_W0_triplet, 1.0);
    // V0 rhs W2
    this->F_0(this->_V0_rhs_W2_triplet, 0.5 * this->_DT);
    // V0 rhs U0
    this->F_0(this->_V0_rhs_U0_triplet, -0.5 * this->_DT);
    // V0 lhs
    this->I(this->_V0_lhs_triplet, 1.0);

    // V1 rhs V0
    this->I(this->_V1_rhs_V0_triplet, 1.0);
    // V1 rhs U0
    this->F_1(this->_V1_rhs_U0_triplet, -0.5 * this->_DT);
    // V1 lhs
    this->I(this->_V1_lhs_triplet, 1.0);
    this->F_1(this->_V1_lhs_triplet, -0.5 * this->_DT);
    
    // V2 rhs V1
    this->I(this->_V2_rhs_V1_triplet, 1.0);
    // V2 rhs U0
    this->F_2(this->_V2_rhs_U0_triplet, -0.5 * this->_DT);
    // V2 lhs
    this->I(this->_V2_lhs_triplet, 1.0);
    this->F_2(this->_V2_lhs_triplet, -0.5 * this->_DT);

    // Now general assignment
    this->_W0_rhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W0_rhs.setFromTriplets(_W0_rhs_triplet.begin(), _W0_rhs_triplet.end());
    _W0_rhs_triplet.clear();
    this->_W0_lhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W0_lhs.setFromTriplets(_W0_lhs_triplet.begin(), _W0_lhs_triplet.end());
    _W0_lhs_triplet.clear();

    this->_W1_rhs_W0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W1_rhs_W0.setFromTriplets(_W1_rhs_W0_triplet.begin(), _W1_rhs_W0_triplet.end());
    _W1_rhs_W0_triplet.clear();
    this->_W1_rhs_U0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W1_rhs_U0.setFromTriplets(_W1_rhs_U0_triplet.begin(), _W1_rhs_U0_triplet.end());
    _W1_rhs_U0_triplet.clear();
    this->_W1_lhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W1_lhs.setFromTriplets(_W1_lhs_triplet.begin(), _W1_lhs_triplet.end());
    _W1_lhs_triplet.clear();
    
    this->_W2_rhs_W1 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W2_rhs_W1.setFromTriplets(_W2_rhs_W1_triplet.begin(), _W2_rhs_W1_triplet.end());
    _W2_rhs_W1_triplet.clear();
    this->_W2_rhs_U0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W2_rhs_U0.setFromTriplets(_W2_rhs_U0_triplet.begin(), _W2_rhs_U0_triplet.end());
    _W2_rhs_U0_triplet.clear();
    this->_W2_lhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _W2_lhs.setFromTriplets(_W2_lhs_triplet.begin(), _W2_lhs_triplet.end());
    _W2_lhs_triplet.clear();
    
    this->_V0_rhs_W0 = SpMat(this->_NX * this->_NX, this->_NY * this->_NY);
    _V0_rhs_W0.setFromTriplets(_V0_rhs_W0_triplet.begin(), _V0_rhs_W0_triplet.end());
    _V0_rhs_W0_triplet.clear();
    this->_V0_rhs_W2 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V0_rhs_W2.setFromTriplets(_V0_rhs_W2_triplet.begin(), _V0_rhs_W2_triplet.end());
    _V0_rhs_W2_triplet.clear();
    this->_V0_rhs_U0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V0_rhs_U0.setFromTriplets(_V0_rhs_U0_triplet.begin(), _V0_rhs_U0_triplet.end());
    _V0_rhs_U0_triplet.clear();
    this->_V0_lhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V0_lhs.setFromTriplets(_V0_lhs_triplet.begin(), _V0_lhs_triplet.end());
    _V0_lhs_triplet.clear();
    
    this->_V1_rhs_V0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V1_rhs_V0.setFromTriplets(_V1_rhs_V0_triplet.begin(), _V1_rhs_V0_triplet.end());
    _V1_rhs_V0_triplet.clear();
    this->_V1_rhs_U0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V1_rhs_U0.setFromTriplets(_V1_rhs_U0_triplet.begin(), _V1_rhs_U0_triplet.end());
    _V1_rhs_U0_triplet.clear();
    this->_V1_lhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V1_lhs.setFromTriplets(_V1_lhs_triplet.begin(), _V1_lhs_triplet.end());
    _V1_lhs_triplet.clear();
    
    this->_V2_rhs_V1 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V2_rhs_V1.setFromTriplets(_V2_rhs_V1_triplet.begin(), _V2_rhs_V1_triplet.end());
    _V2_rhs_V1_triplet.clear();
    this->_V2_rhs_U0 = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V2_rhs_U0.setFromTriplets(_V2_rhs_U0_triplet.begin(), _V2_rhs_U0_triplet.end());
    _V2_rhs_U0_triplet.clear();
    this->_V2_lhs = SpMat(this->_NX * this->_NY, this->_NX * this->_NY);
    _V2_lhs.setFromTriplets(_V2_lhs_triplet.begin(), _V2_lhs_triplet.end());
    _V2_lhs_triplet.clear();

    this->_W1_solver_it.compute(this->_W1_lhs);
    this->_W2_solver_it.compute(this->_W2_lhs);
    this->_V1_solver_it.compute(this->_V1_lhs);
    this->_V2_solver_it.compute(this->_V2_lhs);

    //this->_W1_solver.analyzePattern(this->_W1_lhs);
    //this->_W1_solver.factorize(this->_W1_lhs);
    //this->_W2_solver.analyzePattern(this->_W2_lhs);
    //this->_W2_solver.factorize(this->_W2_lhs);
    //this->_V1_solver.analyzePattern(this->_V1_lhs);
    //this->_V1_solver.factorize(this->_V1_lhs);
    //this->_V2_solver.analyzePattern(this->_V2_lhs);
    //this->_V2_solver.factorize(this->_V2_lhs);

    /*
    std::cout << _W0_rhs << std::endl;
    std::cout << _W0_rhs << std::endl;
    std::cout << _W0_lhs << std::endl;
    std::cout << _W1_rhs_W0 << std::endl;
    std::cout << _W1_rhs_U0 << std::endl;
    std::cout<< _W1_lhs<<std::endl;
    std::cout << _W2_rhs_W1 << std::endl;
    std::cout<< _W2_rhs_U0<<std::endl;
    std::cout << _W2_lhs << std::endl;
    std::cout << _V0_rhs_W0 << std::endl;
    std::cout << _V0_rhs_W2 << std::endl;
    std::cout<< _V0_rhs_U0<<std::endl;
    std::cout << _V0_lhs << std::endl;
    std::cout<< _V1_rhs_V0<<std::endl;
    std::cout << _V1_rhs_U0 << std::endl;
    std::cout<< _V1_lhs<<std::endl;
    std::cout << _V2_rhs_V1 << std::endl;
    std::cout<< _V2_rhs_U0<<std::endl;
    std::cout << _V2_lhs << std::endl;/* */
}


// GETTERS
const size_t &crank_nicolson_2d::NX() const
{
    return this->_NX;
}
const size_t &crank_nicolson_2d::NY() const
{
    return this->_NY;
}
const double &crank_nicolson_2d::XMIN() const
{
    return this->_XMIN;
}
const double &crank_nicolson_2d::XMAX() const
{
    return this->_XMAX;
}
const double &crank_nicolson_2d::YMIN() const
{
    return this->_YMIN;
}
const double &crank_nicolson_2d::YMAX() const
{
    return this->_YMAX;
}
const double &crank_nicolson_2d::DT() const
{
    return this->_DT;
}
const double &crank_nicolson_2d::DX() const
{
    return this->_DX;
}
const double &crank_nicolson_2d::DY() const
{
    return this->_DY;
}

const MatrixXdr &crank_nicolson_2d::U0_mat() const
{
    return this->_U0_mat;
}
const MatrixXdr &crank_nicolson_2d::U_mat() const
{
    return this->_U_mat;
}
const MatrixXdr &crank_nicolson_2d::C_x() const
{
    return this->_C_x;
}
const MatrixXdr &crank_nicolson_2d::C_y() const
{
    return this->_C_y;
}
const MatrixXdr &crank_nicolson_2d::A_xx() const
{
    return this->_A_xx;
}
const MatrixXdr &crank_nicolson_2d::A_yy() const
{
    return this->_A_yy;
}
const MatrixXdr &crank_nicolson_2d::A_xy() const
{
    return this->_A_xy;
}

const unsigned int &crank_nicolson_2d::executed_iterations() const
{
    return this->_executed_iterations;
}
