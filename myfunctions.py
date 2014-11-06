#encoding=utf-8


import os
import warnings
import numpy as np
import time
from scipy import sparse
from scipy import linalg
from scipy.sparse import coo_matrix
import pylab as pl
from sklearn.metrics import r2_score
import sklearn.linear_model
import logging
from optparse import OptionParser
import sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import density
from sklearn import metrics
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from sklearn.utils import arrayfuncs,array2d,check_random_state, as_float_array, ConvergenceWarning
import math
import sklearn.decomposition
import gensim
from gensim import corpora, models, similarities

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from sklearn.linear_model.base import LinearModel,BaseEstimator
##from sklearn.linear_model import lars_path,LassoCV
from sklearn.utils import arrayfuncs,array2d,check_random_state, as_float_array, ConvergenceWarning
from sklearn.datasets.samples_generator import make_regression
import warnings
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.six.moves import xrange
from sklearn.decomposition import dict_learning,sparse_encode
from numpy.lib.stride_tricks import as_strided
from sklearn.base import RegressorMixin,TransformerMixin
import csv
import time
import sys
import random
import math
import sklearn
from math import log
from distutils.version import LooseVersion
from scipy import linalg, interpolate
from scipy.linalg.lapack import get_lapack_funcs
import scipy
import math
import sklearn.preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


solve_triangular_args = {}
if LooseVersion(scipy.__version__) >= LooseVersion('0.12'):
    solve_triangular_args = {'check_finite': False}


def lars_path(ori_Gram, ori_Cov, n_samples,Xy=None, Gram=None, max_iter=500,
              alpha_min=0, method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=True):

    n_features = ori_Gram.shape[1]
    n_samples = n_samples
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    # We are initializing this to "zeros" and not empty, because
    # it is passed to scipy linalg functions and thus if it has NaNs,
    # even if they are in the upper part that it not used, we
    # get errors raised.
    # Once we support only scipy > 0.12 we can use check_finite=False and
    # go back to "empty"
    L = np.zeros((max_features, max_features), dtype=Gram.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Gram,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (Gram,))

    if Gram is None:
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')
    elif Gram == 'auto':
        Gram = None
        if X.shape[0] > X.shape[1]:
            Gram = ori_Gram
##            ori_Gram=np.dot(X.T, X)
    elif copy_Gram:
            Gram = ori_Gram.copy()

    if Xy is None:
        Cov = ori_Cov
##        ori_Cov=np.dot(X.T, y)
    else:
        Cov = ori_Cov

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning

    while True:
        if Cov.size:
            C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
##            print C_
            C = np.fabs(C_)
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

##        alpha[0] = C / n_samples
        alpha[0] = C 
        if alpha[0] <= alpha_min:  # early stopping
            if not (abs(alpha[0] - alpha_min) < tiny):
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **solve_triangular_args)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)
                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny))
        gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                alphas = np.resize(alphas, n_iter + add_features)
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
            alpha = alphas[n_iter, np.newaxis]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            # handle the case when idx is not length of 1
            [arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii) for ii in
                idx]

            n_active -= 1
            m, n = idx, n_active
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]

            if Gram is None:
                # propagate dropped variable
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        # yeah this is stupid
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]

                # TODO: this could be updated
##                residual = y - np.dot(X[:, :n_active], coef[active])
##                temp = np.dot(X.T[n_active], residual)
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)

                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i+1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                          Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
##                residual = y - np.dot(X, coef)
##                temp = np.dot(X.T[drop_idx], residual)
##                Cov = np.r_[temp, Cov]
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)
                Cov=np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp)))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        return alphas, active, coefs.T
    else:
        return alpha, active, coef

def lars_path_lasso(ori_Gram, ori_Cov, n_samples,Xy=None, Gram=None, max_iter=500,
              alpha_min=0, method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=True):

    n_features = ori_Gram.shape[1]
    n_samples = n_samples
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    # We are initializing this to "zeros" and not empty, because
    # it is passed to scipy linalg functions and thus if it has NaNs,
    # even if they are in the upper part that it not used, we
    # get errors raised.
    # Once we support only scipy > 0.12 we can use check_finite=False and
    # go back to "empty"
    L = np.zeros((max_features, max_features), dtype=Gram.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Gram,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (Gram,))

    if Gram is None:
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')
    elif Gram == 'auto':
        Gram = None
        if X.shape[0] > X.shape[1]:
            Gram = ori_Gram
##            ori_Gram=np.dot(X.T, X)
    elif copy_Gram:
            Gram = ori_Gram.copy()

    if Xy is None:
        Cov = ori_Cov
##        ori_Cov=np.dot(X.T, y)
    else:
        Cov = ori_Cov

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning

    while True:
        if Cov.size:
            C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
##            print C_
            C = np.fabs(C_)
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

        alpha[0] = C / n_samples
##        alpha[0] = C 
        if alpha[0] <= alpha_min:  # early stopping
            if not (abs(alpha[0] - alpha_min) < tiny):
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **solve_triangular_args)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)
                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny))
        gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                alphas = np.resize(alphas, n_iter + add_features)
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
            alpha = alphas[n_iter, np.newaxis]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            # handle the case when idx is not length of 1
            [arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii) for ii in
                idx]

            n_active -= 1
            m, n = idx, n_active
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]

            if Gram is None:
                # propagate dropped variable
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        # yeah this is stupid
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]

                # TODO: this could be updated
##                residual = y - np.dot(X[:, :n_active], coef[active])
##                temp = np.dot(X.T[n_active], residual)
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)

                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i+1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                          Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
##                residual = y - np.dot(X, coef)
##                temp = np.dot(X.T[drop_idx], residual)
##                Cov = np.r_[temp, Cov]
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)
                Cov=np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp)))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        return alphas, active, coefs.T
    else:
        return alpha, active, coef




def eles_path(ori_Gram, ori_Cov, n_samples,alpha,l1_ratio,Xy=None, Gram=None, max_iter=500,
               method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=True):
    l1=alpha*l1_ratio
    l2=alpha*(1-l1_ratio)

    
    ga_ma=l2/(l2+1)
    
    ori_Gram=(1-ga_ma)*ori_Gram+np.diag([ga_ma/n_samples]*len(ori_Gram))

    

    n_features = ori_Gram.shape[1]
    n_samples = n_samples
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    # We are initializing this to "zeros" and not empty, because
    # it is passed to scipy linalg functions and thus if it has NaNs,
    # even if they are in the upper part that it not used, we
    # get errors raised.
    # Once we support only scipy > 0.12 we can use check_finite=False and
    # go back to "empty"
    L = np.zeros((max_features, max_features), dtype=Gram.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Gram,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (Gram,))

    if Gram is None:
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')
    elif Gram == 'auto':
        Gram = None
        if X.shape[0] > X.shape[1]:
            Gram = ori_Gram
##            ori_Gram=np.dot(X.T, X)
    elif copy_Gram:
            Gram = ori_Gram.copy()

    if Xy is None:
        Cov = ori_Cov
##        ori_Cov=np.dot(X.T, y)
    else:
        Cov = ori_Cov

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning

    while True:
        if Cov.size:
            C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
##            print C_
            C = np.fabs(C_)
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

##        alpha[0] = C / n_samples
        alpha[0] = C 
        if alpha[0] <= alpha_min:  # early stopping
            if not (abs(alpha[0] - alpha_min) < tiny):
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **solve_triangular_args)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)
                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny))
        gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                alphas = np.resize(alphas, n_iter + add_features)
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
            alpha = alphas[n_iter, np.newaxis]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            # handle the case when idx is not length of 1
            [arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii) for ii in
                idx]

            n_active -= 1
            m, n = idx, n_active
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]

            if Gram is None:
                # propagate dropped variable
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        # yeah this is stupid
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]

                # TODO: this could be updated
##                residual = y - np.dot(X[:, :n_active], coef[active])
##                temp = np.dot(X.T[n_active], residual)
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)

                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i+1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                          Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
##                residual = y - np.dot(X, coef)
##                temp = np.dot(X.T[drop_idx], residual)
##                Cov = np.r_[temp, Cov]
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)
                Cov=np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp)))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        return alphas, active, coefs.T
    else:
        return alpha, active, coef

def eles_path_lasso(ori_Gram, ori_Cov, n_samples,alpha,l1_ratio,Xy=None, Gram=None, max_iter=500,
               method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=True):
    l1=alpha*l1_ratio
    l2=alpha*(1-l1_ratio)

    
    ga_ma=l2/(l2+1)
    
    ori_Gram=(1-ga_ma)*ori_Gram+np.diag([ga_ma/n_samples]*len(ori_Gram))

    

    n_features = ori_Gram.shape[1]
    n_samples = n_samples
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    # We are initializing this to "zeros" and not empty, because
    # it is passed to scipy linalg functions and thus if it has NaNs,
    # even if they are in the upper part that it not used, we
    # get errors raised.
    # Once we support only scipy > 0.12 we can use check_finite=False and
    # go back to "empty"
    L = np.zeros((max_features, max_features), dtype=Gram.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Gram,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (Gram,))

    if Gram is None:
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')
    elif Gram == 'auto':
        Gram = None
        if X.shape[0] > X.shape[1]:
            Gram = ori_Gram
##            ori_Gram=np.dot(X.T, X)
    elif copy_Gram:
            Gram = ori_Gram.copy()

    if Xy is None:
        Cov = ori_Cov
##        ori_Cov=np.dot(X.T, y)
    else:
        Cov = ori_Cov

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning

    while True:
        if Cov.size:
            C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
##            print C_
            C = np.fabs(C_)
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

        alpha[0] = C / n_samples
##        alpha[0] = C 
        if alpha[0] <= alpha_min:  # early stopping
            if not (abs(alpha[0] - alpha_min) < tiny):
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **solve_triangular_args)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)
                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny))
        gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                alphas = np.resize(alphas, n_iter + add_features)
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
            alpha = alphas[n_iter, np.newaxis]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            # handle the case when idx is not length of 1
            [arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii) for ii in
                idx]

            n_active -= 1
            m, n = idx, n_active
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]

            if Gram is None:
                # propagate dropped variable
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        # yeah this is stupid
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]

                # TODO: this could be updated
##                residual = y - np.dot(X[:, :n_active], coef[active])
##                temp = np.dot(X.T[n_active], residual)
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)

                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i+1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                          Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
##                residual = y - np.dot(X, coef)
##                temp = np.dot(X.T[drop_idx], residual)
##                Cov = np.r_[temp, Cov]
                temp=ori_Cov[drop_idx]-np.dot(Gram[drop_idx],coef)
                Cov=np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp)))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        return alphas, active, coefs.T
    else:
        return alpha, active, coef



############### lasso & elestic Net covariance matrix #####################

def lasso(ori_Gram, ori_Cov, n_samples,alpha,Xy=None, Gram=None, max_iter=500,
               method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=False):
    alpha,active, coef=lars_path_lasso(ori_Gram, ori_Cov, n_samples,Xy=Xy, Gram=Gram, max_iter=max_iter,alpha_min=alpha, method='lar', copy_X=True,\
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=False)
    return alpha,active, coef


def elesticNet(ori_Gram, ori_Cov, n_samples,alpha,l1_ratio, Xy=None, Gram=None, max_iter=500,
               method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=False):
    alpha,active, coef=eles_path_lasso(ori_Gram, ori_Cov, n_samples,alpha,l1_ratio,Xy=Xy, Gram=Gram,  max_iter=500,
              method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=False)
    return alpha,active, coef

def spca(gram,ori_Gram,n_samples,n_features,n_targets,alpha,max_iter=1000):
    ii=0
    n=n_samples
    [l,b]=np.linalg.eig(gram)
    a=b[:,0:n_targets]
    cov=np.dot(gram,a)
    cov=np.array(cov)
    errors=[]
    for ii in range(max_iter): 
        coef= np.empty((n_targets,n_features))
        for k in xrange(n_targets):
            xy=np.array(cov[:,k])
            ori_xy=np.array(cov[:,k]) 
            res = lars_path(ori_Gram,ori_xy, n_samples,Xy=xy, Gram=gram, max_iter=max_iter,alpha_min=alpha, method='lars', copy_X=True,\
                      eps=np.finfo(np.float).eps,\
                      copy_Gram=True, verbose=0, return_path=False)
            coef[k]=res[2]
        b=coef.T

        UU, DD, VV = linalg.svd(np.dot(ori_Gram,b), full_matrices=False)
        a=np.dot(UU,VV)
        cov = np.dot(ori_Gram,a)
        
        current_cost=n*(np.abs(np.trace(gram)+np.trace(np.dot(np.dot(coef,gram),coef.T)-\
                                2*np.dot(cov.T,coef.T))))+ alpha * np.sum(np.abs(coef))
        
        errors.append(current_cost)
        if ii > 0:
            dE = errors[-2] - errors[-1]

            if dE < np.finfo(np.float).eps *errors[-1]:
##                print("--- Convergence reached after %d iterations" % ii)
                
                break          
##    print 'penalty parameter '+str(alpha)
    b=b/np.linalg.norm(b)
    return a,b


def spca_bic(alpha_min,alpha_max,inc,gram,ori_Gram,n_samples,n_features,\
             n_targets,max_iter=1000):
    alphas=[]
    alphas.append(alpha_min)
    while alpha_min<alpha_max:
        alpha_min+=inc
        alphas.append(alpha_min)
    bic_list=[]
    b_list=[]
    ev_list=[]
    for alpha in alphas:
        a,b=spca(gram,ori_Gram,n_samples,n_features,n_targets,alpha,max_iter=max_iter)
        non_coef=0
        for i in np.transpose(b):
            for ii in i:
                if abs(ii)>0.000001:
                    non_coef+=1
        bic_=BIC(a,b,ori_Gram,non_coef,n_samples)
        ev=PEV(b,gram)
        bic_list.append(bic_)
        b_list.append(b)
        ev_list.append(ev)
    
    ind=bic_list.index(min(bic_list))
    choose_alpha=alphas[ind]
    choose_b=b_list[ind]
    choose_bic=bic_list[ind]
    choose_ev=ev_list[ind]
       
     
    return choose_alpha,choose_b,choose_bic,choose_ev


def gn(X):
    n_samples,m_features=X.shape
    meanV=np.mean(X,axis=0)
    return meanV,n_samples,m_features

def I_gn(pre_mean,pre_n,New_X):
    New_n=New_X.shape[0]+pre_n
    m_new=New_X.shape[1]
    mean=(pre_n*pre_mean+np.sum(New_X,axis=0))/New_n
    return mean,New_n,m_new

def W_(X):
    W=np.dot(X.T,X)
    return W

def I_W(W,New_X):
    New_w=W+np.dot(New_X.T,New_X)
    return New_w


def V_(W,N,mean):
    V=W/N-np.dot(mean.T,mean)
    return V

def V2R(V):
    dia=1.0/np.diag(V)
##    print dia
    ss=[s**(0.50) for s in dia]
    A=np.diag(ss)
##    print A
    R=np.dot(A,np.dot(V,A))
##    print R
##    print np.std(R,axis=0)
##    print sum(R)
    return R
    
def fXX(X,Y):
    return np.dot(np.transpose(X),Y)
    
def inverseX(X):
    return np.linalg.inv(X)

def xishu(invXX,XY):
    return np.dot(invXX,XY)    


def PEV(b,G):
    Var=np.dot(np.dot(np.transpose(b),G),b)
    Var_diag=np.diag(Var)
##    print '**************'
##    print Var_diag
    Var_sum=sum(Var_diag)
##    print Var_sum
    percent=[]
    for i in Var_diag:
        percent.append(i/Var_sum)
    return Var_sum
##    
##    return Var_diag, percent

def Ajust_PEV(b,G):
    Var=np.dot(np.dot(np.transpose(b),G),b)
    eig= np.linalg.eig(Var)
    EV=sum(eig[0])
    return EV

def Normb_PEV(b,G):
    b_n=b/np.linalg.norm(b)
    Var=np.dot(np.dot(np.transpose(b_n),G),b_n)
    eig= np.linalg.eig(Var)
    EV=sum(eig[0])
    return EV


def BIC(a,b,G,non_coef,n):
    bic=np.trace(np.dot(np.dot((a-b).T,G),(a-b)))+non_coef*log(n)/n
    return bic


def Ajust_BIC(a,b,G,non_coef,n):
    bic=sum(np.linalg.eig(np.dot(np.dot((a-b).T,G),(a-b)))[0])+non_coef*log(n)/n
    return bic



def gensim2coo(name):
    record=0
    row=[]
    column=[]
    value=[]
    classy=[]
    with open(name) as f:
        for l in f:
            data=l.rstrip().split()
            
            
            classy.append(int(data[0]))
            for c in data[1:]:
                pair=c.split(':')
                row.append(record)
                column.append(int(pair[0]))
                value.append(float(pair[1]))
                
            record+=1

    X_coo=sparse.coo_matrix((value,(row,column)))
##    X=sparse.coo_matrix((value,(row,column)),shape=(max(row+1),max(column+1))).todense()
    y=classy
    return X_coo,y

def load_csv(f):
    f=csv.reader(open(f,'r'))
    y=[]
    temp=[]
    x=[]
    p=0
    for l in f:
        y.append(float(l[0]))
        for j in l[1:]:
            temp.append(float(j))
        x.append(temp)
        temp=[]
        p+=1
        
    print 'done the read data' ,p
    return x,y
def load_csv_x(f):
    f=csv.reader(open(f,'r'))
    ll=[]
    X=[]
    p=0
    for l in f:
        try:
            ll.append(float(l[0]))
        except ValueError:
            print l
            continue
        else:
            for lll in l[1:]:
                ll.append(float(lll))
            X.append(ll)
            ll=[]
            p+=1
    print 'done the read data',p
    return X

    

def write_line(num,per,line):
    f=open(str(per)+'1th.txt','a')
    line=str(per)+' '+str(num)+' '+line[:-1]
    f.write(line+'\n')
    f.close()
    


def sample_yx2x_y(num,per,y_x):
    yx_o=random.sample(y_x,num)
    yx_add=random.sample(y_x,int(math.ceil(num*per)))
    yx_all=yx_o[:]
    for i in yx_add:
        yx_all.append(i)
    x_o=array2d(yx_o)[:,1:]
    y_o=array2d(yx_o)[:,0]
    x_add=array2d(yx_add)[:,1:]
    y_add=array2d(yx_add)[:,0]
    x_all=array2d(yx_all)[:,1:]
    y_all=array2d(yx_all)[:,0]
    return x_o,y_o,x_add,y_add,x_all,y_all

def x2gram(x_o):
    mean_o,n_o,m_o=gn(x_o)
    w_o=W_(x_o)
    v_o=V_(w_o,n_o,mean_o)
    R_o=V2R(v_o)
    gram_o=R_o
    ori_Gram_o=R_o
    return mean_o,n_o,w_o,m_o,gram_o,ori_Gram_o


def x2gram_inc(mean_o,n_o,w_o,x_add):
    mean_new,n_new,m_new=I_gn(mean_o,n_o,x_add)
    w_new=I_W(w_o,x_add)
    v_new=V_(w_new,n_new,mean_new)
    R_new=V2R(v_new)
    gram_new=R_new
    ori_Gram_new=R_new
    return mean_new,n_new,w_new,m_new,gram_new,ori_Gram_new

def x2gram_lasso(x_o):
    mean_o,n_o,m_o=gn(x_o)
    w_o=W_(x_o)
    gram_o=w_o
    ori_Gram_o=w_o
    return mean_o,n_o,w_o,m_o,gram_o,ori_Gram_o


def x2gram_inc_lasso(mean_o,n_o,w_o,x_add):
    mean_new,n_new,m_new=I_gn(mean_o,n_o,x_add)
    w_new=I_W(w_o,x_add)
    gram_new=w_new
    ori_Gram_new=w_new
    return mean_new,n_new,w_new,m_new,gram_new,ori_Gram_new

def fXX(X,Y):
    return np.dot(np.transpose(X),Y)
    
def inverseX(X):
    return np.linalg.inv(X)

def xishu(invXX,XY):
    return np.dot(invXX,XY)

def increase_xishu(XX,XY,X1,Y1):
    increase_XX=fXX(X1,X1)
    increase_XY=fXX(X1,Y1)
    new_XX=XX+increase_XX
    new_XY=XY+increase_XY
    xishu_r=xishu(new_XX,new_XY)
    return xishu_r
def SSE(YY,YX,xishu):
    SSE=YY-YX*xishu
    return SSE

def SST(YY,XY,num):
    SST=YY-(XY[0])*(XY[0])/num
    return SST

def Fmeasure(SSE,SST,p,num):
    Fm=((SST-SSE)/p)/(SSE/(num-p-1))
    return Fm

def tmeasure(B,SSE,invXX,num,p):
    diag= np.diag(invXX)
    Sb=SSE/(num-p-1)*diag
    T=[B[i]/np.transpose(Sb)[i] for i in range(0,4)]
    return T
