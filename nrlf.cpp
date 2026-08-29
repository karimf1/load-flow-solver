// nrlf -- Newton-Raphson load flow solver.
//
// The whole project is this one file: types, Y-bus assembly, the Newton-Raphson
// solver, branch flows, SVG plotting, the two test networks, and the full test
// suite. Build it with a single command and no build system:
//
//     g++ -std=c++17 -O2 -I/path/to/eigen3 nrlf.cpp -o nrlf
//
//     ./nrlf case14 --flows     solve and print the branch flow table
//     ./nrlf --test             run all 193 checks
//
// Reading order below: types, Y-bus, solver, flows, plotting, test networks,
// then the tests, then main().

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===== types.hpp =========================================================
namespace nrlf {

using Complex = std::complex<double>;

/// PQ    — load bus:      P, Q known;    |V|, theta unknown
/// PV    — generator bus: P, |V| known;  Q,   theta unknown
/// Slack — reference bus: |V|, theta known; P, Q unknown (one per case)
enum class BusType { PQ, PV, Slack };

inline const char* to_string(BusType t) {
  switch (t) {
    case BusType::PQ:    return "PQ";
    case BusType::PV:    return "PV";
    case BusType::Slack: return "Slack";
  }
  return "?";
}

/// All quantities are per-unit on the case baseMVA.
struct Bus {
  int id = 0;                 ///< bus number as written in the case data
  BusType type = BusType::PQ;

  double Pd = 0.0, Qd = 0.0;  ///< demand
  double Pg = 0.0, Qg = 0.0;  ///< generation
  double Gs = 0.0, Bs = 0.0;  ///< shunt to ground (+Bs = capacitive)
  double Vm = 1.0;            ///< voltage magnitude, p.u.
  double Va = 0.0;            ///< voltage angle, radians

  double Pinj() const { return Pg - Pd; }
  double Qinj() const { return Qg - Qd; }
};

/// A transmission line, or a transformer when `tap` is set. Modelled as an
/// equivalent pi section behind an ideal tap changer on the `from` end:
///
///      from        1:a                      to
///        o----[]----+---[ y = 1/(r+jx) ]----+----o
///                   |                       |
///                 jb/2                    jb/2
struct Branch {
  int from = 0, to = 0;
  double r = 0.0, x = 0.0;   ///< series impedance, p.u.
  double b = 0.0;            ///< TOTAL charging susceptance (b/2 stamped per end)
  double tap = 0.0;          ///< off-nominal tap ratio; 0 means nominal (1.0)

  /// Series admittance y = 1/(r + jx).
  Complex y_series() const {
    const Complex z(r, x);
    if (z == Complex(0.0, 0.0)) {
      throw std::invalid_argument("branch " + std::to_string(from) + "-" +
                                  std::to_string(to) + " has zero impedance");
    }
    return Complex(1.0, 0.0) / z;
  }

  double ratio() const { return tap == 0.0 ? 1.0 : tap; }
};

struct Case {
  std::string name;
  double baseMVA = 100.0;
  std::vector<Bus> buses;
  std::vector<Branch> branches;

  std::size_t nbus() const { return buses.size(); }

  /// Map a bus number to its row/column index in Y. Linear search is fine at
  /// the sizes this solver targets.
  int index_of(int bus_id) const {
    for (std::size_t i = 0; i < buses.size(); ++i) {
      if (buses[i].id == bus_id) return static_cast<int>(i);
    }
    throw std::out_of_range("branch references undefined bus " +
                            std::to_string(bus_id));
  }

  int slack_index() const {
    for (std::size_t i = 0; i < buses.size(); ++i) {
      if (buses[i].type == BusType::Slack) return static_cast<int>(i);
    }
    throw std::invalid_argument("case '" + name + "' has no slack bus");
  }
};

}  // namespace nrlf

// ===== ybus.hpp ==========================================================
namespace nrlf {

using MatrixXcd = Eigen::MatrixXcd;

/// The 2x2 admittance of a single branch, relating the currents entering it at
/// each end to the voltages there:
///
///     [ I_f ]   [ ff  ft ] [ V_f ]
///     [ I_t ] = [ tf  tt ] [ V_t ]
///
/// With y = 1/(r+jx), charging b and tap ratio a:
///
///     tt = y + jb/2        ff = (y + jb/2) / a^2        ft = tf = -y / a
///
/// At a = 1 this is the ordinary line model. Both Y-bus assembly and the
/// branch flow calculation go through here, so the two cannot disagree.
struct BranchY {
  Complex ff, ft, tf, tt;
};

BranchY branch_admittance(const Branch& br);

/// Build the bus admittance matrix Y, where I = Y V. Each branch adds its
/// BranchY entries at the (from, to) rows and columns; each bus adds its own
/// shunt Gs + jBs to the diagonal.
MatrixXcd build_ybus(const Case& c);

/// Print Y as a list of nonzero entries (readable at any size).
void print_ybus(std::ostream& os, const MatrixXcd& Y, const Case& c);

}  // namespace nrlf

// ===== powerflow.hpp =====================================================
namespace nrlf {

struct PFOptions {
  double tol = 1e-8;    ///< converged when max |mismatch| falls below this, p.u.
  int max_iter = 20;
  bool verbose = false; ///< print the mismatch at each iteration
};

struct PFResult {
  bool converged = false;
  int iterations = 0;
  std::vector<double> history;  ///< max |mismatch| before each Newton update
  std::vector<double> Vm, Va;   ///< solution, indexed like Case::buses (Va in radians)
};

/// Real and reactive power injected into the network at each bus, for a given
/// voltage state:
///
///   P_i = V_i * sum_k V_k (G_ik cos(t_i - t_k) + B_ik sin(t_i - t_k))
///   Q_i = V_i * sum_k V_k (G_ik sin(t_i - t_k) - B_ik cos(t_i - t_k))
void injections(const MatrixXcd& Y, const std::vector<double>& Vm,
                const std::vector<double>& Va, std::vector<double>& P,
                std::vector<double>& Q);

/// Solve the power flow by Newton-Raphson, starting from a flat start.
PFResult solve_powerflow(const Case& c, const PFOptions& opt = PFOptions{});

/// Bus voltage table, slack injection, and the convergence history.
void print_solution(std::ostream& os, const Case& c, const PFResult& r);

}  // namespace nrlf

// ===== flows.hpp =========================================================
namespace nrlf {

/// Complex power flowing in a branch at the converged solution, in per-unit.
/// Both quantities are measured *into* the branch, so their sum is what the
/// branch consumes:
///
///     S_from = V_f conj(I_f),  S_to = V_t conj(I_t),  loss = S_from + S_to
///
/// loss.real() is the I^2 R heating and is always positive. loss.imag() is the
/// series I^2 X absorption net of the line charging, and can be negative on a
/// lightly loaded line whose charging dominates.
struct BranchFlow {
  int from = 0, to = 0;
  Complex S_from, S_to, loss;
};

std::vector<BranchFlow> branch_flows(const Case& c, const PFResult& r);

/// Per-branch flow table, in MW / MVAr on the case base.
void print_flows(std::ostream& os, const Case& c,
                 const std::vector<BranchFlow>& flows);

}  // namespace nrlf

// ===== plot.hpp ==========================================================
namespace nrlf {

/// Write standalone SVG figures. SVG is generated directly rather than shelling
/// out to a plotting library, so the repo stays dependency-free and the figures
/// render inline on GitHub.
void write_voltage_profile(const std::string& path, const Case& c,
                           const PFResult& r);

void write_convergence(const std::string& path, const Case& c,
                       const PFResult& r);

}  // namespace nrlf

// ===== cases.hpp =========================================================
namespace nrlf {

/// Three buses in a triangle with a capacitor on bus 3. Small enough to check
/// the whole Y matrix by hand.
Case make_case3();

/// IEEE 14-bus test system: 5 generators, 20 branches, 3 tap transformers,
/// and a shunt capacitor on bus 9. Line data is the standard published set.
Case make_case14();

}  // namespace nrlf

// ===== ybus.cpp ==========================================================
namespace nrlf {

BranchY branch_admittance(const Branch& br) {
  const Complex y = br.y_series();
  const double a = br.ratio();
  const Complex tt = y + Complex(0.0, br.b / 2.0);
  return {tt / (a * a), -y / a, -y / a, tt};
}

MatrixXcd build_ybus(const Case& c) {
  const auto n = static_cast<Eigen::Index>(c.nbus());
  MatrixXcd Y = MatrixXcd::Zero(n, n);

  // Bus shunts land on the diagonal.
  for (Eigen::Index i = 0; i < n; ++i) {
    const Bus& bus = c.buses[static_cast<std::size_t>(i)];
    Y(i, i) += Complex(bus.Gs, bus.Bs);
  }

  for (const Branch& br : c.branches) {
    const auto f = static_cast<Eigen::Index>(c.index_of(br.from));
    const auto t = static_cast<Eigen::Index>(c.index_of(br.to));
    if (f == t) {
      throw std::invalid_argument("branch from bus " + std::to_string(br.from) +
                                  " to itself");
    }

    const BranchY by = branch_admittance(br);

    // += rather than = so parallel branches accumulate.
    Y(f, f) += by.ff;
    Y(f, t) += by.ft;
    Y(t, f) += by.tf;
    Y(t, t) += by.tt;
  }

  return Y;
}

void print_ybus(std::ostream& os, const MatrixXcd& Y, const Case& c) {
  os << "Y-bus (" << Y.rows() << " x " << Y.rows() << ")\n";
  os << "  bus  bus                 Y(i,j)\n";
  os << "  ---------------------------------------\n";
  os << std::fixed << std::setprecision(4);

  for (Eigen::Index i = 0; i < Y.rows(); ++i) {
    for (Eigen::Index j = 0; j < Y.cols(); ++j) {
      const Complex z = Y(i, j);
      if (std::abs(z) < 1e-12) continue;
      os << std::setw(5) << c.buses[static_cast<std::size_t>(i)].id
         << std::setw(5) << c.buses[static_cast<std::size_t>(j)].id
         << std::setw(12) << z.real() << (z.imag() < 0 ? " - " : " + ")
         << std::setw(9) << std::fabs(z.imag()) << "j\n";
    }
  }
  os << "\n";
}

}  // namespace nrlf

// ===== powerflow.cpp =====================================================
namespace nrlf {
namespace {

/// The four partial derivatives of (P_i, Q_i) with respect to (theta_k, V_k).
///
/// Off-diagonal (i != k), writing t = theta_i - theta_k:
///   dP_i/dt_k = V_i V_k (G sin t - B cos t)
///   dP_i/dV_k = V_i       (G cos t + B sin t)
///   dQ_i/dt_k = -V_i V_k (G cos t + B sin t)
///   dQ_i/dV_k = V_i       (G sin t - B cos t)
///
/// Diagonal (i == k), simplified using the injections P_i and Q_i themselves,
/// which removes the summation:
///   dP_i/dt_i = -Q_i - B_ii V_i^2
///   dP_i/dV_i =  P_i/V_i + G_ii V_i
///   dQ_i/dt_i =  P_i - G_ii V_i^2
///   dQ_i/dV_i =  Q_i/V_i - B_ii V_i
struct Partials {
  double dPdt, dPdV, dQdt, dQdV;
};

Partials partials(int i, int k, const MatrixXcd& Y, const std::vector<double>& Vm,
                  const std::vector<double>& Va, const std::vector<double>& P,
                  const std::vector<double>& Q) {
  const double G = Y(i, k).real();
  const double B = Y(i, k).imag();

  if (i == k) {
    return {-Q[i] - B * Vm[i] * Vm[i],
             P[i] / Vm[i] + G * Vm[i],
             P[i] - G * Vm[i] * Vm[i],
             Q[i] / Vm[i] - B * Vm[i]};
  }

  const double t = Va[i] - Va[k];
  const double ct = std::cos(t), st = std::sin(t);
  const double vv = Vm[i] * Vm[k];
  return {vv * (G * st - B * ct),
          Vm[i] * (G * ct + B * st),
          -vv * (G * ct + B * st),
          Vm[i] * (G * st - B * ct)};
}

}  // namespace

void injections(const MatrixXcd& Y, const std::vector<double>& Vm,
                const std::vector<double>& Va, std::vector<double>& P,
                std::vector<double>& Q) {
  const int n = static_cast<int>(Vm.size());
  P.assign(static_cast<std::size_t>(n), 0.0);
  Q.assign(static_cast<std::size_t>(n), 0.0);

  for (int i = 0; i < n; ++i) {
    double p = 0.0, q = 0.0;
    for (int k = 0; k < n; ++k) {
      const double G = Y(i, k).real();
      const double B = Y(i, k).imag();
      const double t = Va[static_cast<std::size_t>(i)] - Va[static_cast<std::size_t>(k)];
      const double ct = std::cos(t), st = std::sin(t);
      p += Vm[static_cast<std::size_t>(k)] * (G * ct + B * st);
      q += Vm[static_cast<std::size_t>(k)] * (G * st - B * ct);
    }
    P[static_cast<std::size_t>(i)] = Vm[static_cast<std::size_t>(i)] * p;
    Q[static_cast<std::size_t>(i)] = Vm[static_cast<std::size_t>(i)] * q;
  }
}

PFResult solve_powerflow(const Case& c, const PFOptions& opt) {
  const int n = static_cast<int>(c.nbus());
  const MatrixXcd Y = build_ybus(c);
  c.slack_index();  // throws if the case has no reference bus

  // Unknowns: the angle of every non-slack bus, and the magnitude of every
  // PQ bus. Any consistent ordering works; these two index lists define it.
  std::vector<int> ang, mag;
  for (int i = 0; i < n; ++i) {
    const BusType t = c.buses[static_cast<std::size_t>(i)].type;
    if (t != BusType::Slack) ang.push_back(i);
    if (t == BusType::PQ) mag.push_back(i);
  }
  const int na = static_cast<int>(ang.size());
  const int nm = static_cast<int>(mag.size());
  const int nx = na + nm;

  PFResult r;
  r.Vm.resize(static_cast<std::size_t>(n));
  r.Va.resize(static_cast<std::size_t>(n));

  // Flat start: PQ buses at 1.0 p.u., slack and PV buses hold their scheduled
  // magnitude; all angles at zero.
  for (int i = 0; i < n; ++i) {
    const Bus& b = c.buses[static_cast<std::size_t>(i)];
    r.Vm[static_cast<std::size_t>(i)] = (b.type == BusType::PQ) ? 1.0 : b.Vm;
    r.Va[static_cast<std::size_t>(i)] = 0.0;
  }

  if (nx == 0) {  // a lone slack bus is already solved
    r.converged = true;
    return r;
  }

  std::vector<double> P, Q;
  Eigen::VectorXd mismatch(nx);
  Eigen::MatrixXd J(nx, nx);

  for (int iter = 0;; ++iter) {
    injections(Y, r.Vm, r.Va, P, Q);

    // Mismatch = scheduled - calculated. Real power at every non-slack bus,
    // reactive power at the PQ buses only (Q is a free variable at PV buses).
    for (int a = 0; a < na; ++a) {
      const int i = ang[static_cast<std::size_t>(a)];
      mismatch(a) = c.buses[static_cast<std::size_t>(i)].Pinj() - P[static_cast<std::size_t>(i)];
    }
    for (int a = 0; a < nm; ++a) {
      const int i = mag[static_cast<std::size_t>(a)];
      mismatch(na + a) = c.buses[static_cast<std::size_t>(i)].Qinj() - Q[static_cast<std::size_t>(i)];
    }

    const double err = mismatch.cwiseAbs().maxCoeff();
    r.history.push_back(err);
    if (opt.verbose) {
      std::printf("  iter %2d   max mismatch = %.3e\n", iter, err);
    }
    if (err < opt.tol) {
      r.converged = true;
      break;
    }
    if (iter >= opt.max_iter) break;

    // Jacobian, four blocks:  J = [ dP/dtheta  dP/dV ]
    //                             [ dQ/dtheta  dQ/dV ]
    for (int a = 0; a < na; ++a) {
      const int i = ang[static_cast<std::size_t>(a)];
      for (int b = 0; b < na; ++b) {
        const int k = ang[static_cast<std::size_t>(b)];
        J(a, b) = partials(i, k, Y, r.Vm, r.Va, P, Q).dPdt;
      }
      for (int b = 0; b < nm; ++b) {
        const int k = mag[static_cast<std::size_t>(b)];
        J(a, na + b) = partials(i, k, Y, r.Vm, r.Va, P, Q).dPdV;
      }
    }
    for (int a = 0; a < nm; ++a) {
      const int i = mag[static_cast<std::size_t>(a)];
      for (int b = 0; b < na; ++b) {
        const int k = ang[static_cast<std::size_t>(b)];
        J(na + a, b) = partials(i, k, Y, r.Vm, r.Va, P, Q).dQdt;
      }
      for (int b = 0; b < nm; ++b) {
        const int k = mag[static_cast<std::size_t>(b)];
        J(na + a, na + b) = partials(i, k, Y, r.Vm, r.Va, P, Q).dQdV;
      }
    }

    const Eigen::VectorXd dx = J.partialPivLu().solve(mismatch);
    if (!dx.allFinite()) {
      throw std::runtime_error(
          "power flow diverged: the Jacobian is singular at iteration " +
          std::to_string(iter) + " (check for an islanded or unsolvable case)");
    }

    for (int a = 0; a < na; ++a) r.Va[static_cast<std::size_t>(ang[static_cast<std::size_t>(a)])] += dx(a);
    for (int a = 0; a < nm; ++a) r.Vm[static_cast<std::size_t>(mag[static_cast<std::size_t>(a)])] += dx(na + a);
    r.iterations = iter + 1;
  }

  return r;
}

void print_solution(std::ostream& os, const Case& c, const PFResult& r) {
  os << (r.converged ? "Converged" : "DID NOT CONVERGE") << " in " << r.iterations
     << " iterations\n\n";

  os << "  iter    max mismatch (p.u.)\n";
  os << "  ---------------------------\n";
  for (std::size_t i = 0; i < r.history.size(); ++i) {
    os << std::setw(6) << i << "    " << std::scientific << std::setprecision(3)
       << r.history[i] << "\n";
  }
  os << "\n";

  std::vector<double> P, Q;
  injections(build_ybus(c), r.Vm, r.Va, P, Q);

  os << "Bus voltages\n";
  os << "  bus   type        |V| (p.u.)   angle (deg)\n";
  os << "  ------------------------------------------\n";
  os << std::fixed;
  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    os << std::setw(5) << c.buses[i].id << "   " << std::left << std::setw(7)
       << to_string(c.buses[i].type) << std::right << std::setprecision(4)
       << std::setw(12) << r.Vm[i] << std::setprecision(3) << std::setw(14)
       << r.Va[i] * 180.0 / M_PI << "\n";
  }
  os << "\n";

  // Summing every bus injection cancels load against generation and leaves
  // what the network itself takes: the real power lost in branch resistance,
  // and the net reactive absorbed (series reactance minus shunts and charging,
  // so this one can legitimately come out negative).
  double ploss = 0.0, qnet = 0.0;
  for (std::size_t i = 0; i < P.size(); ++i) { ploss += P[i]; qnet += Q[i]; }

  const auto s = static_cast<std::size_t>(c.slack_index());
  os << std::setprecision(2)
     << "Slack bus " << c.buses[s].id << " injection:  "
     << P[s] * c.baseMVA << " MW, " << Q[s] * c.baseMVA << " MVAr\n"
     << "Real power losses:        " << ploss * c.baseMVA << " MW\n"
     << "Net reactive absorbed:    " << qnet * c.baseMVA << " MVAr\n\n";
}

}  // namespace nrlf

// ===== flows.cpp =========================================================
namespace nrlf {

std::vector<BranchFlow> branch_flows(const Case& c, const PFResult& r) {
  std::vector<BranchFlow> out;
  out.reserve(c.branches.size());

  for (const Branch& br : c.branches) {
    const auto f = static_cast<std::size_t>(c.index_of(br.from));
    const auto t = static_cast<std::size_t>(c.index_of(br.to));

    const Complex Vf = std::polar(r.Vm[f], r.Va[f]);
    const Complex Vt = std::polar(r.Vm[t], r.Va[t]);

    const BranchY by = branch_admittance(br);
    const Complex If = by.ff * Vf + by.ft * Vt;
    const Complex It = by.tf * Vf + by.tt * Vt;

    BranchFlow bf;
    bf.from = br.from;
    bf.to = br.to;
    bf.S_from = Vf * std::conj(If);
    bf.S_to = Vt * std::conj(It);
    bf.loss = bf.S_from + bf.S_to;
    out.push_back(bf);
  }
  return out;
}

namespace {
/// Snap values that round to zero, so a lossless transformer prints "0.00"
/// rather than "-0.00".
double tidy(double v) { return (std::fabs(v) < 5e-3) ? 0.0 : v; }
}  // namespace

void print_flows(std::ostream& os, const Case& c,
                 const std::vector<BranchFlow>& flows) {
  const double base = c.baseMVA;

  os << "Branch flows\n";
  os << "   from     to        P (MW)   Q (MVAr)      loss P    loss Q\n";
  os << "  ----------------------------------------------------------------\n";
  os << std::fixed << std::setprecision(2);

  Complex total(0.0, 0.0);
  for (const BranchFlow& b : flows) {
    total += b.loss;
    os << std::setw(7) << b.from << std::setw(7) << b.to
       << std::setw(14) << tidy(b.S_from.real() * base)
       << std::setw(11) << tidy(b.S_from.imag() * base)
       << std::setw(12) << tidy(b.loss.real() * base)
       << std::setw(10) << tidy(b.loss.imag() * base) << "\n";
  }

  os << "  ----------------------------------------------------------------\n";
  os << "  total branch losses" << std::setw(35) << total.real() * base
     << std::setw(10) << total.imag() * base << "\n\n";
}

}  // namespace nrlf

// ===== plot.cpp ==========================================================
namespace nrlf {
namespace {

constexpr double W = 760, H = 400;
constexpr double L = 66, R = 26, T = 46, B = 54;
constexpr double PW = W - L - R;
constexpr double PH = H - T - B;

const char* kBg     = "#fbfbfd";
const char* kFrame  = "#c9ced6";
const char* kGrid   = "#e6e9ee";
const char* kText    = "#24292f";
const char* kMuted  = "#5b6472";
const char* kLine   = "#8a94a6";

const char* color_for(BusType t) {
  switch (t) {
    case BusType::Slack: return "#d64545";
    case BusType::PV:    return "#2f7ed8";
    case BusType::PQ:    return "#3f9a52";
  }
  return kLine;
}

std::string num(double v, int prec) {
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(prec);
  s << v;
  return s.str();
}

void header(std::ostringstream& o, const std::string& title,
            const std::string& subtitle) {
  o << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H
    << "' viewBox='0 0 " << W << " " << H << "' font-family='-apple-system,"
    << "Segoe UI,Helvetica,Arial,sans-serif'>\n"
    << "<rect width='" << W << "' height='" << H << "' fill='" << kBg
    << "' stroke='" << kFrame << "'/>\n"
    << "<text x='" << L << "' y='26' font-size='15' font-weight='600' fill='"
    << kText << "'>" << title << "</text>\n"
    << "<text x='" << (W - R) << "' y='26' font-size='11.5' text-anchor='end' fill='"
    << kMuted << "'>" << subtitle << "</text>\n";
}

void frame(std::ostringstream& o) {
  o << "<rect x='" << L << "' y='" << T << "' width='" << PW << "' height='" << PH
    << "' fill='none' stroke='" << kFrame << "'/>\n";
}

void write_file(const std::string& path, const std::string& body) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << body;
}

}  // namespace

void write_voltage_profile(const std::string& path, const Case& c,
                           const PFResult& r) {
  const int n = static_cast<int>(c.nbus());
  if (n < 2) throw std::runtime_error("voltage profile needs at least 2 buses");

  double lo = r.Vm[0], hi = r.Vm[0];
  for (double v : r.Vm) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
  // Round outward to a 0.02 grid so the axis labels stay tidy.
  lo = std::floor((lo - 0.005) / 0.02) * 0.02;
  hi = std::ceil((hi + 0.005) / 0.02) * 0.02;

  auto px = [&](int i) { return L + PW * i / (n - 1); };
  auto py = [&](double v) { return T + PH * (hi - v) / (hi - lo); };

  std::ostringstream o;
  header(o, "Bus voltage profile &#8212; " + c.name,
         "converged in " + std::to_string(r.iterations) + " iterations");

  // Horizontal gridlines, one per 0.02 p.u.
  for (double v = lo; v <= hi + 1e-9; v += 0.02) {
    const double y = py(v);
    o << "<line x1='" << L << "' y1='" << y << "' x2='" << (L + PW) << "' y2='" << y
      << "' stroke='" << kGrid << "'/>\n"
      << "<text x='" << (L - 10) << "' y='" << (y + 4)
      << "' font-size='11' text-anchor='end' fill='" << kMuted << "'>" << num(v, 2)
      << "</text>\n";
  }

  // Nominal 1.0 p.u. reference.
  if (lo < 1.0 && hi > 1.0) {
    o << "<line x1='" << L << "' y1='" << py(1.0) << "' x2='" << (L + PW) << "' y2='"
      << py(1.0) << "' stroke='" << kMuted << "' stroke-dasharray='4 3'/>\n";
  }

  frame(o);

  o << "<polyline fill='none' stroke='" << kLine << "' stroke-width='1.5' points='";
  for (int i = 0; i < n; ++i) o << px(i) << "," << py(r.Vm[static_cast<std::size_t>(i)]) << " ";
  o << "'/>\n";

  for (int i = 0; i < n; ++i) {
    const Bus& b = c.buses[static_cast<std::size_t>(i)];
    o << "<circle cx='" << px(i) << "' cy='" << py(r.Vm[static_cast<std::size_t>(i)])
      << "' r='4.5' fill='" << color_for(b.type) << "' stroke='" << kBg
      << "' stroke-width='1.5'/>\n"
      << "<text x='" << px(i) << "' y='" << (T + PH + 18)
      << "' font-size='10.5' text-anchor='middle' fill='" << kMuted << "'>" << b.id
      << "</text>\n";
  }

  o << "<text x='" << (L + PW / 2) << "' y='" << (H - 14)
    << "' font-size='11.5' text-anchor='middle' fill='" << kMuted << "'>bus</text>\n"
    << "<text transform='translate(18," << (T + PH / 2) << ") rotate(-90)'"
    << " font-size='11.5' text-anchor='middle' fill='" << kMuted
    << "'>|V| (p.u.)</text>\n";

  // Legend.
  const BusType kinds[3] = {BusType::Slack, BusType::PV, BusType::PQ};
  double lx = L + PW - 150;
  for (const BusType k : kinds) {
    o << "<circle cx='" << lx << "' cy='" << (T + 16) << "' r='4.5' fill='"
      << color_for(k) << "'/>\n"
      << "<text x='" << (lx + 9) << "' y='" << (T + 20) << "' font-size='11' fill='"
      << kMuted << "'>" << to_string(k) << "</text>\n";
    lx += 50;
  }

  o << "</svg>\n";
  write_file(path, o.str());
}

void write_convergence(const std::string& path, const Case& c, const PFResult& r) {
  const int n = static_cast<int>(r.history.size());
  if (n < 2) throw std::runtime_error("convergence plot needs at least 2 points");

  // Log scale: work in decades, clamped so a converged-to-machine-precision
  // final point does not stretch the axis indefinitely.
  auto lg = [](double v) { return std::log10(std::fmax(v, 1e-18)); };
  double lo = lg(r.history[0]), hi = lo;
  for (double v : r.history) { lo = std::fmin(lo, lg(v)); hi = std::fmax(hi, lg(v)); }
  lo = std::floor(lo);
  hi = std::ceil(hi);

  auto px = [&](int i) { return L + PW * i / (n - 1); };
  auto py = [&](double v) { return T + PH * (hi - lg(v)) / (hi - lo); };

  std::ostringstream o;
  header(o, "Newton-Raphson convergence &#8212; " + c.name,
         "quadratic: the error exponent roughly doubles each step");

  // One gridline per decade, thinned out if the range is tall.
  const int decades = static_cast<int>(hi - lo);
  const int step = (decades > 10) ? 3 : 1;
  for (int d = 0; d <= decades; d += step) {
    const double y = T + PH * d / decades;
    o << "<line x1='" << L << "' y1='" << y << "' x2='" << (L + PW) << "' y2='" << y
      << "' stroke='" << kGrid << "'/>\n"
      << "<text x='" << (L - 10) << "' y='" << (y + 4)
      << "' font-size='11' text-anchor='end' fill='" << kMuted << "'>1e"
      << static_cast<int>(hi - d) << "</text>\n";
  }

  // Tolerance line.
  const double tol = 1e-8;
  if (lg(tol) > lo && lg(tol) < hi) {
    o << "<line x1='" << L << "' y1='" << py(tol) << "' x2='" << (L + PW) << "' y2='"
      << py(tol) << "' stroke='#d64545' stroke-dasharray='4 3' opacity='0.7'/>\n"
      << "<text x='" << (L + PW - 6) << "' y='" << (py(tol) - 6)
      << "' font-size='10.5' text-anchor='end' fill='#d64545'>tolerance 1e-8</text>\n";
  }

  frame(o);

  o << "<polyline fill='none' stroke='#2f7ed8' stroke-width='2' points='";
  for (int i = 0; i < n; ++i) o << px(i) << "," << py(r.history[static_cast<std::size_t>(i)]) << " ";
  o << "'/>\n";

  for (int i = 0; i < n; ++i) {
    o << "<circle cx='" << px(i) << "' cy='" << py(r.history[static_cast<std::size_t>(i)])
      << "' r='4.5' fill='#2f7ed8' stroke='" << kBg << "' stroke-width='1.5'/>\n"
      << "<text x='" << px(i) << "' y='" << (T + PH + 18)
      << "' font-size='10.5' text-anchor='middle' fill='" << kMuted << "'>" << i
      << "</text>\n";
  }

  o << "<text x='" << (L + PW / 2) << "' y='" << (H - 14)
    << "' font-size='11.5' text-anchor='middle' fill='" << kMuted
    << "'>iteration</text>\n"
    << "<text transform='translate(18," << (T + PH / 2) << ") rotate(-90)'"
    << " font-size='11.5' text-anchor='middle' fill='" << kMuted
    << "'>max power mismatch (p.u.)</text>\n</svg>\n";

  write_file(path, o.str());
}

}  // namespace nrlf

// ===== cases.cpp =========================================================
namespace nrlf {
namespace {

Branch line(int from, int to, double r, double x, double b, double tap = 0.0) {
  Branch br;
  br.from = from; br.to = to;
  br.r = r; br.x = x; br.b = b; br.tap = tap;
  return br;
}

}  // namespace

Case make_case3() {
  Case c;
  c.name = "case3";

  Bus b1; b1.id = 1; b1.type = BusType::Slack;
  Bus b2; b2.id = 2; b2.type = BusType::PV;
  Bus b3; b3.id = 3; b3.type = BusType::PQ; b3.Bs = 0.05;
  c.buses = {b1, b2, b3};

  c.branches = {
      line(1, 2, 0.02,   0.04,  0.02),
      line(1, 3, 0.01,   0.03,  0.00),
      line(2, 3, 0.0125, 0.025, 0.01),
  };
  return c;
}

// IEEE 14-bus. Standard published data, in per-unit on a 100 MVA base.
// Cross-checks: total load sums to 259.0 MW / 73.5 MVAr, and Y(1,1) comes out
// as 6.025 - 19.447j, both of which match the published figures.
Case make_case14() {
  Case c;
  c.name = "IEEE 14-bus";
  c.baseMVA = 100.0;

  // id, type, Pd, Qd, Pg, Vm, Bs   (powers in MW/MVAr, converted below)
  struct Row { int id; BusType type; double Pd, Qd, Pg, Vm, Bs; };
  const Row rows[] = {
      { 1, BusType::Slack,  0.0,  0.0,   0.0, 1.060, 0.0},
      { 2, BusType::PV,    21.7, 12.7,  40.0, 1.045, 0.0},
      { 3, BusType::PV,    94.2, 19.0,   0.0, 1.010, 0.0},
      { 4, BusType::PQ,    47.8, -3.9,   0.0, 1.000, 0.0},
      { 5, BusType::PQ,     7.6,  1.6,   0.0, 1.000, 0.0},
      { 6, BusType::PV,    11.2,  7.5,   0.0, 1.070, 0.0},
      { 7, BusType::PQ,     0.0,  0.0,   0.0, 1.000, 0.0},
      { 8, BusType::PV,     0.0,  0.0,   0.0, 1.090, 0.0},
      { 9, BusType::PQ,    29.5, 16.6,   0.0, 1.000, 19.0},
      {10, BusType::PQ,     9.0,  5.8,   0.0, 1.000, 0.0},
      {11, BusType::PQ,     3.5,  1.8,   0.0, 1.000, 0.0},
      {12, BusType::PQ,     6.1,  1.6,   0.0, 1.000, 0.0},
      {13, BusType::PQ,    13.5,  5.8,   0.0, 1.000, 0.0},
      {14, BusType::PQ,    14.9,  5.0,   0.0, 1.000, 0.0},
  };

  for (const Row& r : rows) {
    Bus b;
    b.id = r.id;
    b.type = r.type;
    b.Pd = r.Pd / c.baseMVA;
    b.Qd = r.Qd / c.baseMVA;
    b.Pg = r.Pg / c.baseMVA;
    b.Bs = r.Bs / c.baseMVA;
    b.Vm = r.Vm;
    c.buses.push_back(b);
  }

  c.branches = {
      line( 1,  2, 0.01938, 0.05917, 0.0528),
      line( 1,  5, 0.05403, 0.22304, 0.0492),
      line( 2,  3, 0.04699, 0.19797, 0.0438),
      line( 2,  4, 0.05811, 0.17632, 0.0340),
      line( 2,  5, 0.05695, 0.17388, 0.0346),
      line( 3,  4, 0.06701, 0.17103, 0.0128),
      line( 4,  5, 0.01335, 0.04211, 0.0),
      line( 4,  7, 0.0,     0.20912, 0.0,    0.978),  // transformer
      line( 4,  9, 0.0,     0.55618, 0.0,    0.969),  // transformer
      line( 5,  6, 0.0,     0.25202, 0.0,    0.932),  // transformer
      line( 6, 11, 0.09498, 0.19890, 0.0),
      line( 6, 12, 0.12291, 0.25581, 0.0),
      line( 6, 13, 0.06615, 0.13027, 0.0),
      line( 7,  8, 0.0,     0.17615, 0.0),
      line( 7,  9, 0.0,     0.11001, 0.0),
      line( 9, 10, 0.03181, 0.08450, 0.0),
      line( 9, 14, 0.12711, 0.27038, 0.0),
      line(10, 11, 0.08205, 0.19207, 0.0),
      line(12, 13, 0.22092, 0.19988, 0.0),
      line(13, 14, 0.17093, 0.34802, 0.0),
  };

  return c;
}

}  // namespace nrlf


// ===========================================================================
// Tests
// ===========================================================================


// ---------------------------------------------------------------------------
// test_ybus.cpp
// ---------------------------------------------------------------------------
#undef CHECK
#undef CHECK_NEAR
#undef CHECK_C

namespace tests_ybus {

using nrlf::Complex;
using nrlf::MatrixXcd;

int checks = 0, failures = 0;

void check(bool ok, const char* what, int line) {
  ++checks;
  if (!ok) { ++failures; std::printf("  FAIL line %d: %s\n", line, what); }
}

void check_c(Complex got, Complex want, const char* what, int line) {
  ++checks;
  if (std::abs(got - want) > 1e-9) {
    ++failures;
    std::printf("  FAIL line %d: %s\n        got %g%+gj, want %g%+gj\n", line,
                what, got.real(), got.imag(), want.real(), want.imag());
  }
}

#define CHECK(cond)      check((cond), #cond, __LINE__)
#define CHECK_C(g, w)    check_c((g), (w), #g " == " #w, __LINE__)

// case3: triangle with a 0.05 p.u. capacitor on bus 3.
//   1-2: z = 0.02+0.04j,    |z|^2 = 0.002      -> y = 10-20j,  b/2 = 0.010
//   1-3: z = 0.01+0.03j,    |z|^2 = 0.001      -> y = 10-30j,  b/2 = 0
//   2-3: z = 0.0125+0.025j, |z|^2 = 0.00078125 -> y = 16-32j,  b/2 = 0.005
void test_case3() {
  std::printf("case3: hand-computed 3x3 Y\n");
  const MatrixXcd Y = nrlf::build_ybus(nrlf::make_case3());

  CHECK(Y.rows() == 3);
  CHECK_C(Y(0, 0), Complex(20.0, -49.990));  // (10-20j+0.01j) + (10-30j)
  CHECK_C(Y(1, 1), Complex(26.0, -51.985));  // (10-20j+0.01j) + (16-32j+0.005j)
  CHECK_C(Y(2, 2), Complex(26.0, -61.945));  // + the 0.05j shunt
  CHECK_C(Y(0, 1), Complex(-10.0, 20.0));
  CHECK_C(Y(0, 2), Complex(-10.0, 30.0));
  CHECK_C(Y(1, 2), Complex(-16.0, 32.0));

  // No phase shifters, so Y is symmetric.
  CHECK((Y - Y.transpose()).cwiseAbs().maxCoeff() < 1e-12);
}

// With no shunts, no charging and no taps, Y is a graph Laplacian: every row
// sums to zero. Shunts to ground are what break this, and what make Y
// invertible.
void test_row_sums() {
  std::printf("row sums vanish without shunts\n");
  nrlf::Case c = nrlf::make_case3();
  for (auto& b : c.buses) { b.Gs = 0.0; b.Bs = 0.0; }
  for (auto& br : c.branches) { br.b = 0.0; }

  const MatrixXcd Y = nrlf::build_ybus(c);
  for (Eigen::Index i = 0; i < Y.rows(); ++i) {
    CHECK_C(Y.row(i).sum(), Complex(0.0, 0.0));
  }
}

// Tap transformer, x = 0.2, a = 0.98:  y = -5j
//   Ytt = -5j,  Yff = -5j/0.9604,  Yft = Ytf = 5j/0.98
// The asymmetry between the two diagonals is the point of the tap model.
void test_tap() {
  std::printf("off-nominal tap stamp\n");
  nrlf::Case c = nrlf::make_case3();
  c.branches = {nrlf::Branch{1, 2, 0.0, 0.2, 0.0, 0.98}};
  c.buses[2].Bs = 0.0;

  const MatrixXcd Y = nrlf::build_ybus(c);
  CHECK_C(Y(0, 0), Complex(0.0, -5.0 / (0.98 * 0.98)));
  CHECK_C(Y(1, 1), Complex(0.0, -5.0));
  CHECK_C(Y(0, 1), Complex(0.0, 5.0 / 0.98));
  CHECK_C(Y(1, 0), Complex(0.0, 5.0 / 0.98));

  // tap = 0 is the "nominal" sentinel and must behave as tap = 1.
  c.branches[0].tap = 0.0;
  const MatrixXcd Yn = nrlf::build_ybus(c);
  CHECK_C(Yn(0, 0), Complex(0.0, -5.0));
  CHECK_C(Yn(0, 1), Complex(0.0, 5.0));
}

// Parallel branches must accumulate, not overwrite.
void test_parallel() {
  std::printf("parallel branches accumulate\n");
  nrlf::Case c = nrlf::make_case3();
  const MatrixXcd Y1 = nrlf::build_ybus(c);
  c.branches.push_back(c.branches[0]);
  const MatrixXcd Y2 = nrlf::build_ybus(c);

  const Complex y12(10.0, -20.0);
  CHECK_C(Y2(0, 1) - Y1(0, 1), -y12);
  CHECK_C(Y2(0, 0) - Y1(0, 0), y12 + Complex(0.0, 0.01));
}

void test_case14() {
  std::printf("IEEE 14-bus\n");
  const nrlf::Case c = nrlf::make_case14();
  const MatrixXcd Y = nrlf::build_ybus(c);

  CHECK(Y.rows() == 14);
  CHECK((Y - Y.transpose()).cwiseAbs().maxCoeff() < 1e-12);

  // Bus 1 touches only buses 2 and 5:
  //   y12 = 1/(0.01938+0.05917j) = 4.99913 - 15.26309j, b/2 = 0.0264
  //   y15 = 1/(0.05403+0.22304j) = 1.02590 -  4.23498j, b/2 = 0.0246
  // Y(1,1) = 6.02503 - 19.44707j, matching the published value 6.025-19.447j.
  CHECK(std::abs(Y(0, 0) - Complex(6.025, -19.447)) < 1e-3);

  // 14 diagonals + 2 entries per branch, no parallel branches in this case.
  int nnz = 0;
  for (Eigen::Index i = 0; i < 14; ++i)
    for (Eigen::Index j = 0; j < 14; ++j)
      if (std::abs(Y(i, j)) > 1e-12) ++nnz;
  CHECK(nnz == 14 + 2 * 20);

  // Load totals are a good check on the transcribed bus data.
  double Pd = 0.0, Qd = 0.0;
  for (const nrlf::Bus& b : c.buses) { Pd += b.Pd; Qd += b.Qd; }
  CHECK(std::abs(Pd * 100.0 - 259.0) < 1e-6);
  CHECK(std::abs(Qd * 100.0 - 73.5) < 1e-6);

  CHECK(c.slack_index() == 0);
}

// Bad input should give a named error, not silent infinities.
void test_errors() {
  std::printf("error diagnostics\n");

  auto throws = [](auto&& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
  };

  nrlf::Case dangling = nrlf::make_case3();
  dangling.branches[0].to = 99;
  CHECK(throws([&] { nrlf::build_ybus(dangling); }));

  nrlf::Case zero_z = nrlf::make_case3();
  zero_z.branches[0].r = 0.0;
  zero_z.branches[0].x = 0.0;
  CHECK(throws([&] { nrlf::build_ybus(zero_z); }));

  nrlf::Case no_slack = nrlf::make_case3();
  no_slack.buses[0].type = nrlf::BusType::PQ;
  CHECK(throws([&] { no_slack.slack_index(); }));
}

// Runs this file's checks. Returns {checks, failures}.
std::pair<int, int> run() {
  test_case3();
  test_row_sums();
  test_tap();
  test_parallel();
  test_case14();
  test_errors();

  std::printf("\n%d checks, %d failures — %s\n", checks, failures,
              failures == 0 ? "PASS" : "FAIL");
  return {checks, failures};
}

}  // namespace tests_ybus


// ---------------------------------------------------------------------------
// test_powerflow.cpp
// ---------------------------------------------------------------------------
#undef CHECK
#undef CHECK_NEAR
#undef CHECK_C

namespace tests_powerflow {

int checks = 0, failures = 0;

void check(bool ok, const char* what, int line) {
  ++checks;
  if (!ok) { ++failures; std::printf("  FAIL line %d: %s\n", line, what); }
}

void check_near(double got, double want, double tol, const char* what, int line) {
  ++checks;
  if (std::fabs(got - want) > tol) {
    ++failures;
    std::printf("  FAIL line %d: %s\n        got %.6f, want %.6f (tol %g)\n",
                line, what, got, want, tol);
  }
}

#define CHECK(cond)               check((cond), #cond, __LINE__)
#define CHECK_NEAR(g, w, tol)     check_near((g), (w), (tol), #g " ~ " #w, __LINE__)

// ---------------------------------------------------------------------------
// Manufactured solution. Pick an arbitrary voltage state, compute the power
// injections it implies, then feed those back in as the scheduled loads. The
// solver must recover the voltages we started from.
//
// This validates the whole pipeline — injections, Jacobian, and the Newton
// loop — against itself, with no external reference data needed. If any
// partial derivative is wrong, Newton either misses this target or fails to
// converge quadratically onto it.
// ---------------------------------------------------------------------------
void test_manufactured_solution() {
  std::printf("manufactured solution: solver recovers a known voltage state\n");

  nrlf::Case c = nrlf::make_case3();
  for (auto& b : c.buses) b.type = nrlf::BusType::PQ;
  c.buses[0].type = nrlf::BusType::Slack;

  // The state we intend to recover.
  const std::vector<double> Vm = {1.06, 0.97, 1.03};
  const std::vector<double> Va = {0.0, -0.08, 0.05};

  std::vector<double> P, Q;
  nrlf::injections(nrlf::build_ybus(c), Vm, Va, P, Q);

  // Schedule exactly those injections as demand (Pinj = Pg - Pd).
  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    c.buses[i].Pd = -P[i];
    c.buses[i].Qd = -Q[i];
  }
  c.buses[0].Vm = Vm[0];  // slack holds its magnitude

  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);

  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    CHECK_NEAR(r.Vm[i], Vm[i], 1e-9);
    CHECK_NEAR(r.Va[i], Va[i], 1e-9);
  }
}

// A PV bus must hold its scheduled magnitude exactly, and the slack bus must
// hold both magnitude and a zero angle.
void test_bus_type_constraints() {
  std::printf("PV buses hold |V|, slack holds |V| and angle\n");

  nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);

  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    if (c.buses[i].type == nrlf::BusType::PQ) continue;
    CHECK_NEAR(r.Vm[i], c.buses[i].Vm, 1e-12);
  }
  CHECK_NEAR(r.Va[static_cast<std::size_t>(c.slack_index())], 0.0, 1e-12);
}

// With no load and no generation, and every controlled bus scheduled at
// 1.0 p.u., the network is unloaded: the solution is exactly flat and lossless.
void test_no_load_is_flat() {
  std::printf("unloaded network solves flat with zero losses\n");

  nrlf::Case c = nrlf::make_case14();
  for (auto& b : c.buses) {
    b.Pd = b.Qd = b.Pg = b.Qg = 0.0;
    b.Bs = 0.0;
    b.Vm = 1.0;
  }
  for (auto& br : c.branches) br.b = 0.0;   // charging would inject reactive power
  for (auto& br : c.branches) br.tap = 0.0; // a tap would force a voltage step

  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);
  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    CHECK_NEAR(r.Vm[i], 1.0, 1e-9);
    CHECK_NEAR(r.Va[i], 0.0, 1e-9);
  }
}

// ---------------------------------------------------------------------------
// IEEE 14-bus against the published solution.
// ---------------------------------------------------------------------------
void test_case14_reference() {
  std::printf("IEEE 14-bus vs published solution\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);

  CHECK(r.converged);
  CHECK(r.iterations <= 5);

  // Published |V| (p.u.) and angle (degrees).
  const double ref_vm[14] = {1.060, 1.045, 1.010, 1.018, 1.020, 1.070, 1.062,
                             1.090, 1.056, 1.051, 1.057, 1.055, 1.050, 1.036};
  const double ref_va[14] = {0.000,  -4.983, -12.725, -10.313, -8.774,
                             -14.221, -13.360, -13.360, -14.939, -15.097,
                             -14.791, -15.076, -15.156, -16.034};

  std::printf("   bus      |V|     ref       dV      ang(deg)     ref      dA\n");
  double max_dv = 0.0, max_da = 0.0;
  for (int i = 0; i < 14; ++i) {
    const double vm = r.Vm[static_cast<std::size_t>(i)];
    const double va = r.Va[static_cast<std::size_t>(i)] * 180.0 / M_PI;
    const double dv = std::fabs(vm - ref_vm[i]);
    const double da = std::fabs(va - ref_va[i]);
    max_dv = std::fmax(max_dv, dv);
    max_da = std::fmax(max_da, da);
    std::printf("   %3d  %8.4f %8.3f %8.5f  %10.3f %8.3f %7.4f\n",
                i + 1, vm, ref_vm[i], dv, va, ref_va[i], da);
  }
  std::printf("   max |dV| = %.2e p.u.,  max |dAngle| = %.2e deg\n", max_dv, max_da);

  // The reference is quoted to 3 decimals, so agreement to within half of the
  // last digit is the most this comparison can resolve.
  CHECK(max_dv < 5e-4);
  CHECK(max_da < 1e-2);
}

// An independent check on the whole solve: the published IEEE 14-bus result has
// the slack bus generating 232.4 MW and the network losing 13.393 MW. Both fall
// out of the converged solution rather than being fitted to.
void test_case14_slack_and_losses() {
  std::printf("IEEE 14-bus slack generation and losses vs published\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);

  std::vector<double> P, Q;
  nrlf::injections(nrlf::build_ybus(c), r.Vm, r.Va, P, Q);

  double losses = 0.0;
  for (double p : P) losses += p;

  const double slack_mw = P[static_cast<std::size_t>(c.slack_index())] * c.baseMVA;
  std::printf("   slack %.3f MW (ref 232.4),  losses %.3f MW (ref 13.393)\n",
              slack_mw, losses * c.baseMVA);

  CHECK_NEAR(slack_mw, 232.4, 0.05);
  CHECK_NEAR(losses * c.baseMVA, 13.393, 0.01);
}

// At the solution the network must balance: total generation equals total load
// plus losses. Summing every bus injection leaves exactly the losses, since
// load and generation cancel.
void test_power_balance() {
  std::printf("power balance: generation = load + losses\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);

  std::vector<double> P, Q;
  nrlf::injections(nrlf::build_ybus(c), r.Vm, r.Va, P, Q);

  double losses = 0.0, sched_load = 0.0, sched_gen = 0.0;
  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    losses += P[i];
    sched_load += c.buses[i].Pd;
    if (c.buses[i].type != nrlf::BusType::Slack) sched_gen += c.buses[i].Pg;
  }
  const std::size_t s = static_cast<std::size_t>(c.slack_index());
  const double slack_p = P[s];

  std::printf("   slack %.2f MW, load %.2f MW, other gen %.2f MW, losses %.2f MW\n",
              slack_p * 100, sched_load * 100, sched_gen * 100, losses * 100);

  CHECK_NEAR(slack_p + sched_gen, sched_load + losses, 1e-9);
  CHECK(losses > 0.0);          // a resistive network must lose real power
  CHECK(losses * 100 < 50.0);   // but not an absurd amount on a 259 MW system

  // Every non-slack bus must actually be sitting at its scheduled real power.
  for (std::size_t i = 0; i < c.buses.size(); ++i) {
    if (c.buses[i].type == nrlf::BusType::Slack) continue;
    CHECK_NEAR(P[i], c.buses[i].Pinj(), 1e-8);
  }
}

// Newton-Raphson converges quadratically: the mismatch exponent roughly doubles
// each step once it is near the solution.
void test_quadratic_convergence() {
  std::printf("mismatch falls quadratically\n");

  const nrlf::PFResult r = nrlf::solve_powerflow(nrlf::make_case14());
  CHECK(r.converged);
  CHECK(r.history.size() >= 3);

  for (std::size_t i = 1; i < r.history.size(); ++i) {
    CHECK(r.history[i] < r.history[i - 1]);
  }
  // Two steps in from the start, the error should already be squaring.
  const double a = r.history[1], b = r.history[2];
  CHECK(b < a * a * 10.0);
}

void test_no_slack_throws() {
  std::printf("a case with no slack bus is rejected\n");

  nrlf::Case c = nrlf::make_case14();
  for (auto& b : c.buses) {
    if (b.type == nrlf::BusType::Slack) b.type = nrlf::BusType::PV;
  }
  bool threw = false;
  try { nrlf::solve_powerflow(c); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
}

// Runs this file's checks. Returns {checks, failures}.
std::pair<int, int> run() {
  test_manufactured_solution();
  test_bus_type_constraints();
  test_no_load_is_flat();
  test_case14_reference();
  test_case14_slack_and_losses();
  test_power_balance();
  test_quadratic_convergence();
  test_no_slack_throws();

  std::printf("\n%d checks, %d failures — %s\n", checks, failures,
              failures == 0 ? "PASS" : "FAIL");
  return {checks, failures};
}

}  // namespace tests_powerflow


// ---------------------------------------------------------------------------
// test_flows.cpp
// ---------------------------------------------------------------------------
#undef CHECK
#undef CHECK_NEAR
#undef CHECK_C

namespace tests_flows {

using nrlf::Complex;

int checks = 0, failures = 0;

void check(bool ok, const char* what, int line) {
  ++checks;
  if (!ok) { ++failures; std::printf("  FAIL line %d: %s\n", line, what); }
}

void check_near(double got, double want, double tol, const char* what, int line) {
  ++checks;
  if (std::fabs(got - want) > tol) {
    ++failures;
    std::printf("  FAIL line %d: %s\n        got %.9f, want %.9f (tol %g)\n",
                line, what, got, want, tol);
  }
}

#define CHECK(cond)            check((cond), #cond, __LINE__)
#define CHECK_NEAR(g, w, tol)  check_near((g), (w), (tol), #g " ~ " #w, __LINE__)

/// Power absorbed by a bus shunt: |V|^2 conj(Gs + jBs).
Complex shunt_power(const nrlf::Bus& b, double vm) {
  return vm * vm * std::conj(Complex(b.Gs, b.Bs));
}

// ---------------------------------------------------------------------------
// The central test. At every bus, the power flowing away into the incident
// branches, plus what the local shunt absorbs, must equal the power injected
// there. This is Tellegen / conservation of complex power stated per bus, and
// it ties the branch flow model back to the Y-bus the solver actually used.
// A sign error or a dropped tap term in either place breaks it.
// ---------------------------------------------------------------------------
void test_per_bus_conservation() {
  std::printf("per-bus complex power conservation\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);

  const auto flows = nrlf::branch_flows(c, r);
  std::vector<double> P, Q;
  nrlf::injections(nrlf::build_ybus(c), r.Vm, r.Va, P, Q);

  std::vector<Complex> out(c.nbus(), Complex(0.0, 0.0));
  for (const auto& f : flows) {
    out[static_cast<std::size_t>(c.index_of(f.from))] += f.S_from;
    out[static_cast<std::size_t>(c.index_of(f.to))] += f.S_to;
  }

  for (std::size_t i = 0; i < c.nbus(); ++i) {
    const Complex lhs = out[i] + shunt_power(c.buses[i], r.Vm[i]);
    CHECK_NEAR(lhs.real(), P[i], 1e-9);
    CHECK_NEAR(lhs.imag(), Q[i], 1e-9);
  }
}

// Summing the per-bus identity over the whole system: total losses in the
// branches plus what the shunts take equals the net injected power.
void test_system_loss_accounting() {
  std::printf("system loss accounting\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  const auto flows = nrlf::branch_flows(c, r);

  std::vector<double> P, Q;
  nrlf::injections(nrlf::build_ybus(c), r.Vm, r.Va, P, Q);

  Complex branch_loss(0.0, 0.0), shunt(0.0, 0.0), inj(0.0, 0.0);
  for (const auto& f : flows) branch_loss += f.loss;
  for (std::size_t i = 0; i < c.nbus(); ++i) {
    shunt += shunt_power(c.buses[i], r.Vm[i]);
    inj += Complex(P[i], Q[i]);
  }

  std::printf("   branch loss %.3f MW %+.3f MVAr | shunt %.3f MW %+.3f MVAr"
              " | injected %.3f MW %+.3f MVAr\n",
              branch_loss.real() * 100, branch_loss.imag() * 100,
              shunt.real() * 100, shunt.imag() * 100,
              inj.real() * 100, inj.imag() * 100);

  CHECK_NEAR((branch_loss + shunt).real(), inj.real(), 1e-9);
  CHECK_NEAR((branch_loss + shunt).imag(), inj.imag(), 1e-9);

  // case14 has no resistive shunts, so all real loss is in the branches and
  // must match the published 13.393 MW.
  CHECK_NEAR(shunt.real(), 0.0, 1e-12);
  CHECK_NEAR(branch_loss.real() * 100, 13.393, 0.01);
}

// Real loss in a branch is I^2 R: strictly positive where there is resistance,
// and zero on the lossless tap transformers of case14.
void test_branch_losses_physical() {
  std::printf("branch real losses are I^2 R\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  const auto flows = nrlf::branch_flows(c, r);

  CHECK(flows.size() == c.branches.size());

  for (std::size_t i = 0; i < flows.size(); ++i) {
    const nrlf::Branch& br = c.branches[i];
    const double loss = flows[i].loss.real();
    if (br.r == 0.0) {
      CHECK_NEAR(loss, 0.0, 1e-12);   // pure reactance dissipates nothing
    } else {
      CHECK(loss > 0.0);
    }

    // Cross-check against I^2 R computed straight from the terminal voltages.
    const auto f = static_cast<std::size_t>(c.index_of(br.from));
    const auto t = static_cast<std::size_t>(c.index_of(br.to));
    const Complex Vf = std::polar(r.Vm[f], r.Va[f]) / br.ratio();
    const Complex Vt = std::polar(r.Vm[t], r.Va[t]);
    const Complex Iseries = (Vf - Vt) * br.y_series();
    CHECK_NEAR(loss, std::norm(Iseries) * br.r, 1e-9);
  }
}

// A branch with no charging and unity tap must be symmetric: the flow measured
// from either end differs only by the loss.
void test_simple_line_symmetry() {
  std::printf("lossless-charging line: S_from + S_to == loss\n");

  nrlf::Case c = nrlf::make_case3();
  for (auto& br : c.branches) { br.b = 0.0; br.tap = 0.0; }
  c.buses[1].Pd = 0.4;
  c.buses[1].Qd = 0.1;
  c.buses[1].type = nrlf::BusType::PQ;

  const nrlf::PFResult r = nrlf::solve_powerflow(c);
  CHECK(r.converged);

  for (const auto& f : nrlf::branch_flows(c, r)) {
    CHECK_NEAR((f.S_from + f.S_to).real(), f.loss.real(), 1e-12);
    CHECK(f.loss.real() > 0.0);
  }
}

void test_figures_written() {
  std::printf("SVG figures are written\n");

  const nrlf::Case c = nrlf::make_case14();
  const nrlf::PFResult r = nrlf::solve_powerflow(c);

  const std::string a = "test_vp.svg", b = "test_cv.svg";
  nrlf::write_voltage_profile(a, c, r);
  nrlf::write_convergence(b, c, r);

  for (const std::string& p : {a, b}) {
    std::ifstream f(p);
    CHECK(f.good());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    CHECK(content.size() > 500);
    CHECK(content.rfind("<svg", 0) == 0);
    CHECK(content.find("</svg>") != std::string::npos);
    std::remove(p.c_str());
  }
}

// Runs this file's checks. Returns {checks, failures}.
std::pair<int, int> run() {
  test_per_bus_conservation();
  test_system_loss_accounting();
  test_branch_losses_physical();
  test_simple_line_symmetry();
  test_figures_written();

  std::printf("\n%d checks, %d failures — %s\n", checks, failures,
              failures == 0 ? "PASS" : "FAIL");
  return {checks, failures};
}

}  // namespace tests_flows


// Runs every suite and reports one total.
static int run_all_tests() {
  int checks = 0, failures = 0;
  struct Suite { const char* name; std::pair<int, int> (*fn)(); };
  const Suite suites[] = {
      {"Y-bus",      &tests_ybus::run},
      {"power flow", &tests_powerflow::run},
      {"flows",      &tests_flows::run},
  };
  for (const Suite& s : suites) {
    std::printf("=== %s ===\n", s.name);
    const std::pair<int, int> r = s.fn();
    checks += r.first;
    failures += r.second;
    std::printf("\n");
  }
  std::printf("%d checks, %d failures — %s\n", checks, failures,
              failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}

// ===== main ===============================================================
// nrlf — assemble Y-bus and solve the power flow by Newton-Raphson.
//
//     ./nrlf [case3 | case14] [--ybus] [--flows] [--plot DIR] [-v]



int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--test") return run_all_tests();

  std::string which = "case14";
  std::string plot_dir;
  bool show_ybus = false, show_flows = false;
  nrlf::PFOptions opt;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--ybus")       show_ybus = true;
    else if (a == "--flows") show_flows = true;
    else if (a == "-v")      opt.verbose = true;
    else if (a == "--plot") {
      if (++i >= argc) { std::cerr << "--plot needs a directory\n"; return 2; }
      plot_dir = argv[i];
    }
    else if (a[0] != '-')    which = a;
    else {
      std::cerr << "usage: " << argv[0]
                << " [case3 | case14] [--ybus] [--flows] [--plot DIR] [-v]\n"
                << "       " << argv[0] << " --test\n";
      return 2;
    }
  }

  nrlf::Case c;
  if (which == "case3")       c = nrlf::make_case3();
  else if (which == "case14") c = nrlf::make_case14();
  else {
    std::cerr << "unknown case '" << which << "'\n";
    return 2;
  }

  try {
    std::cout << "=== " << c.name << " ===\n\n";
    if (show_ybus) nrlf::print_ybus(std::cout, nrlf::build_ybus(c), c);

    const nrlf::PFResult r = nrlf::solve_powerflow(c, opt);
    nrlf::print_solution(std::cout, c, r);

    if (show_flows) nrlf::print_flows(std::cout, c, nrlf::branch_flows(c, r));

    if (!plot_dir.empty()) {
      nrlf::write_voltage_profile(plot_dir + "/voltage_profile.svg", c, r);
      nrlf::write_convergence(plot_dir + "/convergence.svg", c, r);
      std::cout << "wrote " << plot_dir << "/voltage_profile.svg and "
                << plot_dir << "/convergence.svg\n";
    }
    return r.converged ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
