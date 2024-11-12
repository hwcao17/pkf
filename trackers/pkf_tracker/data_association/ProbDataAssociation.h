#pragma once

#include "permanent.h"
#include <Eigen/Dense>
#include <vector>

void removeRowCol(const Eigen::MatrixXd matrix, unsigned int rowToRemove, unsigned int colToRemove, Eigen::MatrixXd& result);
void removeRows(const Eigen::MatrixXd matrix, std::vector<size_t> rowsToRemove, Eigen::MatrixXd& result);

Eigen::MatrixXd compute_weights(const Eigen::MatrixXd& probs);
