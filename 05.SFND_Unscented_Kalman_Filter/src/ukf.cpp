#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

// to [-pi, pi]
// https://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
double normalizeAngle(double x) {
    x = std::fmod(x + M_PI, 2 * M_PI);
    if (x < 0) {
        x += 2 * M_PI;
    }
    return x - M_PI;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // set weights
    weights_.setConstant(0.5 / (lambda_ + n_aug_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    // init state covariance matrix
    P_.setIdentity();
    P_(0, 0) = 0.01;
    P_(1, 1) = 0.01;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

    // init state
    if (!is_initialized_) {
        switch (meas_package.sensor_type_) {
        case MeasurementPackage::SensorType::LASER:
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
            break;
        case MeasurementPackage::SensorType::RADAR:
            const double rho = meas_package.raw_measurements_[0];
            const double phi = meas_package.raw_measurements_[1];
            x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
            break;
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }

    // compute the time elapsed between the current and previous measurements
    // dt - expressed in seconds
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    // Do state prediction
    Prediction(dt);

    // Do measurement update
    switch (meas_package.sensor_type_) {
    case MeasurementPackage::SensorType::LASER:
        UpdateLidar(meas_package);
        break;
    case MeasurementPackage::SensorType::RADAR:
        UpdateRadar(meas_package);
        break;
    }
}

void UKF::Prediction(double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */

    // create augmented mean state
    Eigen::Vector<double, 7> x_aug = Eigen::Vector<double, 7>::Zero();
    x_aug.head<5>() = x_;

    // create augmented covariance matrix
    Eigen::Matrix<double, 7, 7> P_aug = Eigen::Matrix<double, 7, 7>::Zero();
    P_aug.topLeftCorner<5, 5>() = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    // calculate square root of P
    Eigen::Matrix<double, 7, 7> L = P_aug.llt().matrixL();

    // create augmented sigma points
    Eigen::Matrix<double, n_aug_, n_sigma_pts_> Xsig_aug;
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    // predict sigma points
    for (int i = 0; i < n_sigma_pts_; ++i) {
        // extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        // add noise
        px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p += nu_a * delta_t;
        yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p += nu_yawdd * delta_t;

        // write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    // predicted state mean
    x_ = (Xsig_pred_.array().rowwise() * weights_.transpose().array()).rowwise().sum();

    // predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_pts_; ++i) { // iterate over sigma points
        Eigen::Vector<double, n_x_> x_diff = Xsig_pred_.col(i) - x_;
        x_diff(3) = normalizeAngle(x_diff(3));

        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
    constexpr int n_z = 2;

    // transform sigma points into measurement space
    Eigen::Matrix<double, n_z, n_sigma_pts_> Zsig;
    for (int i = 0; i < n_sigma_pts_; ++i) { // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    // set measuremnt noise covariance
    Eigen::DiagonalMatrix<double, n_z> R{std_laspx_ * std_laspx_, std_laspy_ * std_laspy_};

    // update measurement
    updateMeasurementCommon<n_z>(Zsig, meas_package.raw_measurements_, R);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */

    constexpr int n_z = 3;

    // transform sigma points into measurement space
    Eigen::Matrix<double, n_z, n_sigma_pts_> Zsig;
    for (int i = 0; i < n_sigma_pts_; ++i) { // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
        Zsig(1, i) = atan2(p_y, p_x);                                     // phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
    }

    // set measuremnt noise covariance
    Eigen::DiagonalMatrix<double, n_z> R(
        std_radr_ * std_radr_,
        std_radphi_ * std_radphi_,
        std_radrd_ * std_radrd_);

    // update measurement
    updateMeasurementCommon<n_z>(Zsig, meas_package.raw_measurements_, R);
}

template <int n_z>
void UKF::updateMeasurementCommon(
    const Eigen::Matrix<double, n_z, n_sigma_pts_> &Zsig,
    const Eigen::Vector<double, n_z> &raw_measurements,
    const Eigen::DiagonalMatrix<double, n_z> &R) {

    // mean predicted measurement
    Eigen::Vector<double, n_z> z_pred =
        (Zsig.array().rowwise() * weights_.transpose().array()).rowwise().sum();

    // innovation covariance matrix S
    Eigen::Matrix<double, n_z, n_z> S = Eigen::Matrix<double, n_z, n_z>::Zero();
    for (int i = 0; i < n_sigma_pts_; ++i) {
        // residual
        Eigen::Vector<double, n_z> z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = normalizeAngle(z_diff(1));

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    S += R.toDenseMatrix();

    // calculate cross correlation matrix
    Eigen::Matrix<double, n_x_, n_z> Tc = Eigen::Matrix<double, n_x_, n_z>::Zero();
    for (int i = 0; i < n_sigma_pts_; ++i) {
        // residual
        Eigen::Vector<double, n_z> z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = normalizeAngle(z_diff(1));

        // state difference
        Eigen::Vector<double, n_x_> x_diff = Xsig_pred_.col(i) - x_;
        x_diff(3) = normalizeAngle(x_diff(3));

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    Eigen::Matrix<double, n_x_, n_z> K = Tc * S.inverse();

    // residual
    Eigen::Vector<double, n_z> z_diff = raw_measurements - z_pred;
    z_diff(1) = normalizeAngle(z_diff(1));

    // update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}
