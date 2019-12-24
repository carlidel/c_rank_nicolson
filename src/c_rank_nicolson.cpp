#include "c_rank_nicolson.h"

// crank-nicolson

// CONSTRUCTORS

crank_nicolson::crank_nicolson()
{
}

crank_nicolson::crank_nicolson(size_t N, double XMIN, double XMAX, double DT, std::vector<double> X0, std::vector<double> A, std::vector<double> B, std::vector<double> C, std::vector<double> D) : _N(N), _XMIN(XMIN), _XMAX(XMAX), _DT(DT), _A(A), _B(B), _C(C), _D(D), _f(X0), _x(X0), _x0(X0)
{
    // There is no source at the beginning (you have to set it up later if you must!)
    this->_is_there_source = false;

    // And no locks!
    this->_lock_left = false;
    this->_lock_right = false;

    // Are the arguments valid?
    assert(X0.size() == N);
    assert(A.size() == 0 || A.size() == N * 2);
    assert(B.size() == 0 || B.size() == N);
    assert(C.size() == 0 || C.size() == N);
    assert(D.size() == 0 || D.size() == N + 2);
    if (A.size() == 0)
        this->_A = std::vector<double>(N * 2, 0.0);
    if (B.size() == 0)
        this->_B = std::vector<double>(N, 0.0);
    if (C.size() == 0)
        this->_C = std::vector<double>(N, 0.0);
    if (D.size() == 0)
        this->_D = std::vector<double>(N + 2, 0.0);
    // Extra Allocation
    this->_f_prime.resize(N);
    this->_c_prime.resize(N);
    this->_temp.resize(N);
    this->_a_left.resize(N);
    this->_b_left.resize(N);
    this->_c_left.resize(N);
    this->_a_right.resize(N);
    this->_b_right.resize(N);
    this->_c_right.resize(N);
    // Initialization
    this->_DX = (XMAX - XMIN) / (N + 1);
    this->_executed_iterations = 0;
    make_left_hand_matrix();
    make_right_hand_matrix();
    make_c_prime();
}

// PUBLIC METHODS
void crank_nicolson::reset()
{
    std::copy(this->_x0.begin(), this->_x0.end(), this->_x.begin());
    this->_executed_iterations = 0;
}

void crank_nicolson::iterate(unsigned int n_iterations)
{
    for (unsigned int i = 0; i < n_iterations; ++i)
    {
        this->_executed_iterations += 1;
        std::swap(this->_x, this->_f);
        dot_product_tridiagonal();
        tridiagonal_solver();
        if (this->_is_there_source)
        {
            apply_source();
        }
    }
}

// PRIVATE METHODS

void crank_nicolson::make_left_hand_matrix()
{
    for (size_t i = 0; i < this->_N; ++i)
    {
        //if (i > 0)
            this->_a_left[i] = 
                + this->_B[i] / (4 * this->_DX)
                - this->_A[i * 2] / (2 * this->_DX * this->_DX)
                - this->_D[i] / (2 * this->_DX * this->_DX);
        //if (i < this->_N - 1)
            this->_c_left[i] = 
                - this->_B[i] / (4 * this->_DX)
                - this->_A[i * 2 + 1] / (2 * this->_DX * this->_DX)
                - this->_D[i + 2] / (2 * this->_DX * this->_DX);
        this->_b_left[i] = 
            + 1 / this->_DT
            - this->_C[i] / 2
            + this->_A[i * 2 + 1] / (2 * this->_DX * this->_DX)
            + this->_A[i * 2] / (2 * this->_DX * this->_DX)
            + this->_D[i + 1] / (this->_DX * this->_DX);
    }
}

void crank_nicolson::make_right_hand_matrix()
{
    for (size_t i = 0; i < this->_N; ++i)
    {
        //if (i > 0)
            this->_a_right[i] = 
                - this->_B[i] / (4 * this->_DX)
                + this->_A[i * 2] / (2 * this->_DX * this->_DX)
                + this->_D[i] / (2 * this->_DX * this->_DX);

        //if (i < this->_N - 1)
            this->_c_right[i] = 
                + this->_B[i] / (4 * this->_DX)
                + this->_A[i * 2 + 1] / (2 * this->_DX * this->_DX)
                + this->_D[i + 2] / (2 * this->_DX * this->_DX);
        this->_b_right[i] = 
            + 1 / this->_DT
            + this->_C[i] / 2
            - this->_A[i * 2 + 1] / (2 * this->_DX * this->_DX)
            - this->_A[i * 2] / (2 * this->_DX * this->_DX)
            - this->_D[i + 1] / (this->_DX * this->_DX);
    }
}

void crank_nicolson::make_c_prime()
{
    this->_c_prime[0] = this->_c_left[0] / this->_b_left[0];
    for (unsigned int i = 1; i < this->_N - 1; i++)
        this->_c_prime[i] = 
            this->_c_left[i] / (this->_b_left[i] - this->_a_left[i] * this->_c_prime[i - 1]);
}

void crank_nicolson::dot_product_tridiagonal()
{
    this->_temp[0] = + this->_b_right[0] * this->_f[0] 
                     + this->_c_right[0] * this->_f[1];
    this->_temp[this->_N - 1] = 
        + this->_a_right[this->_N - 1] * this->_f[this->_N - 2] 
        + this->_b_right[this->_N - 1] * this->_f[this->_N - 1];
    for (size_t i = 1; i < this->_N - 1; ++i)
    {
        this->_temp[i] = + this->_a_right[i] * this->_f[i - 1] 
                         + this->_b_right[i] * this->_f[i] 
                         + this->_c_right[i] * this->_f[i + 1];
    }

    // Left extreme
    if (this->_lock_left)
    {
        this->_temp[0] += this->_left_extreme * (this->_a_right[0] - this->_a_left[0]);
    }

    // Right extreme
    if (this->_lock_right)
    {
        this->_temp[this->_N - 1] += this->_right_extreme * (this->_c_right[this->_N - 1] - this->_c_left[this->_N - 1]);
    }

    std::copy(this->_temp.begin(), this->_temp.end(), this->_f.begin());
}

void crank_nicolson::tridiagonal_solver()
{
    // compute f_prime
    this->_f_prime[0] = this->_f[0] / this->_b_left[0];
    for (unsigned int i = 1; i < this->_N; i++)
        this->_f_prime[i] = 
            (this->_f[i] - this->_a_left[i] * this->_f_prime[i - 1]) 
            / (this->_b_left[i] - this->_a_left[i] * this->_c_prime[i - 1]);

    // solve for last x value
    this->_x[this->_N - 1] = this->_f_prime[this->_N - 1];

    // solve for remaining x values by back substitution
    for (int i = this->_N - 2; i >= 0; i--)
        this->_x[i] = this->_f_prime[i] - this->_c_prime[i] * this->_x[i + 1];
}

void crank_nicolson::apply_source()
{
    std::copy_if(this->_source.begin(), this->_source.end(), this->_x.begin(), [](double i){return i > 0;});
}

// GENERIC TOOLS

std::vector<double> make_coord_vector(double X_MIN, double X_MAX, size_t N)
{
    double DX = (X_MAX - X_MIN) / (N + 1);
    std::vector<double> coords;
    for (size_t i = 0; i < N; ++i)
        coords.push_back(X_MIN + DX * (i));
    return coords;
}

std::vector<double> make_initial_distribution(std::vector<double> x_coord)
{
    // Desired initial distribution
    std::vector<double> x0;
    double constant = 1.0;
    for (size_t i = 0; i < x_coord.size(); ++i)
        x0.push_back(exp(-(std::max(0., x_coord[i]) / constant)));
    //apply_logistic_correction(x0, x_coord);
    return x0;
}

void apply_logistic_correction(std::vector<double> x, std::vector<double> x_coord, const size_t length)
{
    assert(length < x.size());
    double SLOG = (x_coord[1] - x_coord[0]) * 10;
    double XS = (x_coord[x.size() - 1] + x_coord[x.size() - (1 + length)]) / 2;
    for (size_t i = 0; i < x.size(); ++i)
        x[i] /= (1.0 + exp((x_coord[i] - XS) / SLOG));
}

std::vector<double> make_A_distribution(std::vector<double> x_coords, bool normalization)
{   
    double DX = x_coords[1] - x_coords[0];
    double XX0 = 25.00;
    double ETA = 1.51;
    std::vector<double> basic;
    std::vector<double> definitive;
    for (size_t i = 0; i < x_coords.size(); ++i)
    {
        basic.push_back(exp(-2 * pow(XX0 / std::max(0., x_coords[i]), ETA)));
        definitive.push_back(exp(-2 * pow(XX0 / std::max(0., -DX / 2 + x_coords[i]), ETA)));
        definitive.push_back(exp(-2 * pow(XX0 / std::max(0., +DX / 2 + x_coords[i]), ETA)));
    }
    if (normalization)
    {
        double norm = norm_compute(basic, DX);
        for (size_t i = 0; i < definitive.size(); ++i)
            definitive[i] /= norm;
    }
    return definitive;
}

std::vector<double> make_B_distribution(std::vector<double> x_coords, bool normalization)
{
    size_t size = x_coords.size();
    return std::vector<double>(size, 0);
}

std::vector<double> make_C_distribution(std::vector<double> x_coords, bool normalization)
{
    size_t size = x_coords.size();
    return std::vector<double>(size, 0);
}

double norm_compute(std::vector<double> values, double dx)
{
    double area = 0;
    for (size_t i = 1; i < values.size(); ++i)
        area += dx * 0.5 * (values[i] + values[i - 1]);
    return area;
}

// SETTERS

void crank_nicolson::set_executed_iterations(unsigned int value)
{
    this->_executed_iterations = value;
}

void crank_nicolson::set_x_values(std::vector<double> new_x)
{
    std::copy(new_x.begin(), new_x.end(), this->_x.begin());
}

void crank_nicolson::set_source(std::vector<double> source)
{
    this->_is_there_source = true;
    this->_source = source;
}

void crank_nicolson::remove_source()
{
    this->_is_there_source = false;
}

void crank_nicolson::lock_left()
{
    this->_lock_left = true;
    this->_left_extreme = this->_x[0];
}

void crank_nicolson::lock_right()
{
    this->_lock_right = true;
    this->_right_extreme = this->_x.back();
}

void crank_nicolson::unlock_left()
{
    this->_lock_left = false;
}

void crank_nicolson::unlock_right()
{
    this->_lock_right = false;
}

// GETTERS

const size_t crank_nicolson::N() const
{
    return this->_N;
}

const double crank_nicolson::XMIN() const
{
    return this->_XMIN;
}

const double crank_nicolson::XMAX() const
{
    return this->_XMAX;
}

const double crank_nicolson::DT() const
{
    return this->_DT;
}

const double crank_nicolson::DX() const
{
    return this->_DX;
}

const std::vector<double> crank_nicolson::A() const
{
    return this->_A;
}

const std::vector<double> crank_nicolson::B() const
{
    return this->_B;
}

const std::vector<double> crank_nicolson::C() const
{
    return this->_C;
}

const std::vector<double> crank_nicolson::D() const
{
    return this->_D;
}

const std::vector<double> crank_nicolson::a_left() const
{
    return this->_a_left;
}

const std::vector<double> crank_nicolson::b_left() const
{
    return this->_b_left;
}

const std::vector<double> crank_nicolson::c_left() const
{
    return this->_c_left;
}

const std::vector<double> crank_nicolson::a_right() const
{
    return this->_a_right;
}

const std::vector<double> crank_nicolson::b_right() const
{
    return this->_b_right;
}

const std::vector<double> crank_nicolson::c_right() const
{
    return this->_c_right;
}

const std::vector<double> crank_nicolson::f() const
{
    return this->_f;
}

const std::vector<double> crank_nicolson::c_prime() const
{
    return this->_c_prime;
}

const std::vector<double> crank_nicolson::f_prime() const
{
    return this->_f_prime;
}

const std::vector<double> crank_nicolson::x() const
{
    return this->_x;
}

const std::vector<double> crank_nicolson::x0() const
{
    return this->_x0;
}

const unsigned int crank_nicolson::executed_iterations() const
{
    return this->_executed_iterations;
}