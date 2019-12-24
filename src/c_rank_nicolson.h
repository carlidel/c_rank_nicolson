#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <ratio>
#include <chrono>
#include <ctime>

class crank_nicolson
{
private:
    // PARAMETERS
    size_t _N;
    double _XMIN;
    double _XMAX;
    double _DT;
    double _DX;

    bool _is_there_source;
    bool _lock_left;
    bool _lock_right;

    // COEFFICIENT LIST
    std::vector<double> _A;
    std::vector<double> _B;
    std::vector<double> _C;
    std::vector<double> _D;

    // ELEMENTS
    std::vector<double> _a_left;
    std::vector<double> _b_left;
    std::vector<double> _c_left;

    std::vector<double> _a_right;
    std::vector<double> _b_right;
    std::vector<double> _c_right;

    std::vector<double> _f;

    std::vector<double> _c_prime;
    std::vector<double> _f_prime;
    std::vector<double> _temp;

    std::vector<double> _x;
    std::vector<double> _x0;
    std::vector<double> _source;
    
    double _left_extreme;
    double _right_extreme;

    // STATUS
    unsigned int _executed_iterations;

    // UTILIS
    std::chrono::steady_clock::time_point _begin;
    std::chrono::steady_clock::time_point _end;

public:
    // CONSTRUCTORS
    crank_nicolson();
    crank_nicolson(size_t N, double XMIN, double XMAX, double DT, std::vector<double> X0, std::vector<double> A = std::vector<double>(), std::vector<double> B = std::vector<double>(), std::vector<double> C = std::vector<double>(), std::vector<double> D = std::vector<double>());

    // PUBLIC METHODS
    void reset();
    void iterate(unsigned int n_iterations);

    // SETTERS
    void set_executed_iterations(unsigned int value);
    void set_x_values(std::vector<double> new_x);
    void set_source(std::vector<double> source);
    void remove_source();
    void lock_left();
    void lock_right();
    void unlock_left();
    void unlock_right();

    // GETTERS
    const size_t N() const;
    const double XMIN() const;
    const double XMAX() const;
    const double DT() const;
    const double DX() const;
    const std::vector<double> A() const;
    const std::vector<double> B() const;
    const std::vector<double> C() const;
    const std::vector<double> D() const;
    const std::vector<double> a_left() const;
    const std::vector<double> b_left() const;
    const std::vector<double> c_left() const;
    const std::vector<double> a_right() const;
    const std::vector<double> b_right() const;
    const std::vector<double> c_right() const;
    const std::vector<double> f() const;
    const std::vector<double> c_prime() const;
    const std::vector<double> f_prime() const;
    const std::vector<double> x() const;
    const std::vector<double> x0() const;
    const unsigned int executed_iterations() const;

private:
    // PRIVATE METHODS
    void make_left_hand_matrix();
    void make_right_hand_matrix();
    void make_c_prime();
    void dot_product_tridiagonal();
    void tridiagonal_solver();
    void apply_source();
};

// GENERIC TOOLS

std::vector<double> make_coord_vector(double X_MIN, double X_MAX, size_t N);
std::vector<double> make_initial_distribution(std::vector<double> x_coord);
void apply_logistic_correction(std::vector<double> x, std::vector<double> x_coord, const size_t lenght = 50);

// TODO:: Make A structure less "odd"
/* take a look at A implementation and you'll see what I mean...*/
std::vector<double> make_A_distribution(std::vector<double> x_coords, bool normalization=false);

std::vector<double> make_B_distribution(std::vector<double> x_coords, bool normalization=false);
std::vector<double> make_C_distribution(std::vector<double> x_coords, bool normalization=false);

double norm_compute(std::vector<double> values, double dx);