#pragma once

#include "Eigen/Dense"

class MeasurementPackage {
  public:
    long timestamp_;

    enum class SensorType { LASER, RADAR } sensor_type_;

    Eigen::VectorXd raw_measurements_;
};
