#pragma once

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
  public:
    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(MeasurementPackage meas_package);

    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except for init)
    bool use_laser_ = true;

    // if this is false, radar measurements will be ignored (except for init)
    bool use_radar_ = true;

    // State dimension
    static constexpr int n_x_ = 5;

    // Augmented state dimension
    static constexpr int n_aug_ = 7;

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Eigen::Vector<double, n_x_> x_;

    // state covariance matrix
    Eigen::Matrix<double, n_x_, n_x_> P_;

    // # of sigma points
    static constexpr int n_sigma_pts_ = 2 * n_aug_ + 1;

    // Sigma point spreading parameter
    static constexpr double lambda_ = 3 - n_aug_;

    // predicted sigma points matrix
    Eigen::Matrix<double, n_x_, n_sigma_pts_> Xsig_pred_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    static constexpr double std_a_ = 2.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    static constexpr double std_yawdd_ = 1.0;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */
    // Laser measurement noise standard deviation position1 in m
    static constexpr double std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    static constexpr double std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    static constexpr double std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    static constexpr double std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    static constexpr double std_radrd_ = 0.3;
    /**
     * End DO NOT MODIFY section for measurement noise values
     */

    // Weights of sigma points
    Eigen::Vector<double, n_sigma_pts_> weights_;

    // time when the state is true, in us
    long long time_us_;

  private:
    template <int n_z>
    void updateMeasurementCommon(
        const Eigen::Matrix<double, n_z, n_sigma_pts_> &Zsig,
        const Eigen::Vector<double, n_z> &raw_measurements,
        const Eigen::DiagonalMatrix<double, n_z> &R);
};
