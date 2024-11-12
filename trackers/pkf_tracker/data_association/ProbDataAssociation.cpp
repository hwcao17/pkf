
#include "ProbDataAssociation.h"
#include <chrono>


void removeRowCol(const Eigen::MatrixXd matrix, unsigned int rowToRemove, unsigned int colToRemove, Eigen::MatrixXd& result){
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols();

    result.block(0, 0, rowToRemove, colToRemove) = matrix.block(0, 0, rowToRemove, colToRemove);

    result.block(0, colToRemove, rowToRemove, numCols - colToRemove -1) = 
    matrix.block(0, colToRemove+1, rowToRemove, numCols - colToRemove -1);

    result.block(rowToRemove, 0, numRows - rowToRemove - 1, colToRemove) = 
    matrix.block(rowToRemove+1, 0, numRows - rowToRemove - 1, colToRemove);

    result.block(rowToRemove, colToRemove, numRows - rowToRemove - 1, numCols - colToRemove -1) = 
    matrix.block(rowToRemove+1, colToRemove+1, numRows - rowToRemove - 1, numCols - colToRemove -1);
}

void removeRows(const Eigen::MatrixXd matrix, std::vector<size_t> rowsToRemove, Eigen::MatrixXd& result)
{
    size_t numRows = matrix.rows();
    size_t numCols = matrix.cols();

    size_t idx = 0;
    for (size_t i = 0; i < numRows; ++i) {
        if (std::find(rowsToRemove.begin(), rowsToRemove.end(), i) != rowsToRemove.end())
            continue;
        
        result.row(idx) = matrix.row(i);
        idx++;

    }
}


Eigen::MatrixXd compute_weights(const Eigen::MatrixXd& probs) {
    int m = probs.rows();
    int n = probs.cols();

    if (m == 0) {
        return Eigen::MatrixXd::Zero(m, n);
    }

    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m, n);

    if (m == 1) {
        // double overall_perma = permanentFastest(probs);
        // weights.block(0, 0, m, n) = probs * (1 / overall_perma);
        weights.block(0, 0, m, n) = probs;
        return weights;
    }
    
    if (n == 1) {
        weights.block(0, 0, m, n) = probs;
        return weights;
    }

    Eigen::MatrixXd probs_sub = Eigen::MatrixXd::Zero(probs.rows() - 1, probs.cols() - 1);
    
    // double overall_perma = permanentFastest(probs);

    for (size_t i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            removeRowCol(probs, i, j, probs_sub);
            // weights(i, j) = probs(i, j) * permanentFastest(probs_sub) / overall_perma;
            weights(i, j) = probs(i, j) * permanentFastest(probs_sub);
        }
    }

    // weights = weights / (weights.sum() + 1e-16);
    // normalize each row of weights
    for (size_t i = 0; i < weights.rows(); ++i) {
        weights.row(i) = weights.row(i) / (weights.row(i).sum() + 1e-16);
    }

    return weights;
}

