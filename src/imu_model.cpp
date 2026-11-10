#include "navekf/imu_model.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <stdexcept>

namespace navekf {

Mat12 ImuNoise::Qc() const {
  Mat12 q = Mat12::Zero();
  q.diagonal().segment<3>(0).setConstant(sigma_a * sigma_a);
  q.diagonal().segment<3>(3).setConstant(sigma_g * sigma_g);
  q.diagonal().segment<3>(6).setConstant(sigma_ba * sigma_ba);
  q.diagonal().segment<3>(9).setConstant(sigma_bg * sigma_bg);
  return q;
}

Discrete discretize(const Mat15& F, const Mat15& GQG, const double dt,
                    const Discretization method) {
  const Mat15 I = Mat15::Identity();
  Discrete out;
  switch (method) {
    case Discretization::kFirst:
      out.Phi = I + F * dt;
      out.Qd = GQG * dt;
      return out;
    case Discretization::kSecond: {
      const Mat15 FF = F * F;
      out.Phi = I + F * dt + 0.5 * FF * (dt * dt);
      const Mat15 FQ = F * GQG;
      const Mat15 FQFt = FQ * F.transpose();
      out.Qd =
          GQG * dt + 0.5 * (FQ + FQ.transpose()) * (dt * dt) + FQFt * (std::pow(dt, 3.0) / 3.0);
      return out;
    }
    case Discretization::kVanLoan: {
      using Mat30 = Eigen::Matrix<double, 30, 30>;
      Mat30 A = Mat30::Zero();
      A.topLeftCorner<15, 15>() = -F;
      A.topRightCorner<15, 15>() = GQG;
      A.bottomRightCorner<15, 15>() = F.transpose();
      const Mat30 Adt = A * dt;
      const Mat30 B = Adt.exp();
      out.Phi = B.bottomRightCorner<15, 15>().transpose();
      const Mat15 B12 = B.topRightCorner<15, 15>();
      out.Qd = out.Phi * B12;
      return out;
    }
  }
  throw std::invalid_argument("unknown discretization");
}

}  // namespace navekf
