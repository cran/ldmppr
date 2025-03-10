// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// vec_dist
double vec_dist(const NumericVector& x, const NumericVector& y);
RcppExport SEXP _ldmppr_vec_dist(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(vec_dist(x, y));
    return rcpp_result_gen;
END_RCPP
}
// full_product
double full_product(const double xgrid, const double ygrid, const double tgrid, const NumericMatrix& data, const NumericVector& params);
RcppExport SEXP _ldmppr_full_product(SEXP xgridSEXP, SEXP ygridSEXP, SEXP tgridSEXP, SEXP dataSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double >::type xgrid(xgridSEXP);
    Rcpp::traits::input_parameter< const double >::type ygrid(ygridSEXP);
    Rcpp::traits::input_parameter< const double >::type tgrid(tgridSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(full_product(xgrid, ygrid, tgrid, data, params));
    return rcpp_result_gen;
END_RCPP
}
// C_theta2_i
double C_theta2_i(const NumericVector& xgrid, const NumericVector& ygrid, const double tgrid, const NumericMatrix& data, const NumericVector& params, const NumericVector& bounds);
RcppExport SEXP _ldmppr_C_theta2_i(SEXP xgridSEXP, SEXP ygridSEXP, SEXP tgridSEXP, SEXP dataSEXP, SEXP paramsSEXP, SEXP boundsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type xgrid(xgridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type ygrid(ygridSEXP);
    Rcpp::traits::input_parameter< const double >::type tgrid(tgridSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type bounds(boundsSEXP);
    rcpp_result_gen = Rcpp::wrap(C_theta2_i(xgrid, ygrid, tgrid, data, params, bounds));
    return rcpp_result_gen;
END_RCPP
}
// conditional_sum
double conditional_sum(const NumericVector& obs_t, const double eval_t, const NumericVector& y);
RcppExport SEXP _ldmppr_conditional_sum(SEXP obs_tSEXP, SEXP eval_tSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type obs_t(obs_tSEXP);
    Rcpp::traits::input_parameter< const double >::type eval_t(eval_tSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(conditional_sum(obs_t, eval_t, y));
    return rcpp_result_gen;
END_RCPP
}
// conditional_sum_logical
double conditional_sum_logical(const NumericVector& obs_t, const double eval_t, const LogicalVector& y);
RcppExport SEXP _ldmppr_conditional_sum_logical(SEXP obs_tSEXP, SEXP eval_tSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type obs_t(obs_tSEXP);
    Rcpp::traits::input_parameter< const double >::type eval_t(eval_tSEXP);
    Rcpp::traits::input_parameter< const LogicalVector& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(conditional_sum_logical(obs_t, eval_t, y));
    return rcpp_result_gen;
END_RCPP
}
// vec_to_mat_dist
NumericVector vec_to_mat_dist(const NumericVector& eval_u, const NumericVector& x_col, const NumericVector& y_col);
RcppExport SEXP _ldmppr_vec_to_mat_dist(SEXP eval_uSEXP, SEXP x_colSEXP, SEXP y_colSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type eval_u(eval_uSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type x_col(x_colSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y_col(y_colSEXP);
    rcpp_result_gen = Rcpp::wrap(vec_to_mat_dist(eval_u, x_col, y_col));
    return rcpp_result_gen;
END_RCPP
}
// dist_one_dim
NumericVector dist_one_dim(const double eval_t, const NumericVector& obs_t);
RcppExport SEXP _ldmppr_dist_one_dim(SEXP eval_tSEXP, SEXP obs_tSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double >::type eval_t(eval_tSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type obs_t(obs_tSEXP);
    rcpp_result_gen = Rcpp::wrap(dist_one_dim(eval_t, obs_t));
    return rcpp_result_gen;
END_RCPP
}
// part_1_1_full
double part_1_1_full(const NumericMatrix& data, const NumericVector& params);
RcppExport SEXP _ldmppr_part_1_1_full(SEXP dataSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_1_1_full(data, params));
    return rcpp_result_gen;
END_RCPP
}
// part_1_2_full
double part_1_2_full(const NumericMatrix& data, const NumericVector& params);
RcppExport SEXP _ldmppr_part_1_2_full(SEXP dataSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_1_2_full(data, params));
    return rcpp_result_gen;
END_RCPP
}
// part_1_3_full
double part_1_3_full(const NumericVector& xgrid, const NumericVector& ygrid, const NumericVector& tgrid, const NumericMatrix& data, const NumericVector& params, const NumericVector& bounds);
RcppExport SEXP _ldmppr_part_1_3_full(SEXP xgridSEXP, SEXP ygridSEXP, SEXP tgridSEXP, SEXP dataSEXP, SEXP paramsSEXP, SEXP boundsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type xgrid(xgridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type ygrid(ygridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type tgrid(tgridSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type bounds(boundsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_1_3_full(xgrid, ygrid, tgrid, data, params, bounds));
    return rcpp_result_gen;
END_RCPP
}
// part_1_4_full
double part_1_4_full(const NumericMatrix& data, const NumericVector& params);
RcppExport SEXP _ldmppr_part_1_4_full(SEXP dataSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_1_4_full(data, params));
    return rcpp_result_gen;
END_RCPP
}
// part_1_full
double part_1_full(const NumericVector& xgrid, const NumericVector& ygrid, const NumericVector& tgrid, const NumericMatrix& data, const NumericVector& params, const NumericVector& bounds);
RcppExport SEXP _ldmppr_part_1_full(SEXP xgridSEXP, SEXP ygridSEXP, SEXP tgridSEXP, SEXP dataSEXP, SEXP paramsSEXP, SEXP boundsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type xgrid(xgridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type ygrid(ygridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type tgrid(tgridSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type bounds(boundsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_1_full(xgrid, ygrid, tgrid, data, params, bounds));
    return rcpp_result_gen;
END_RCPP
}
// part_2_full
double part_2_full(const NumericVector& xgrid, const NumericVector& ygrid, const NumericVector& tgrid, const NumericMatrix& data, const NumericVector& params, const NumericVector& bounds);
RcppExport SEXP _ldmppr_part_2_full(SEXP xgridSEXP, SEXP ygridSEXP, SEXP tgridSEXP, SEXP dataSEXP, SEXP paramsSEXP, SEXP boundsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type xgrid(xgridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type ygrid(ygridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type tgrid(tgridSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type bounds(boundsSEXP);
    rcpp_result_gen = Rcpp::wrap(part_2_full(xgrid, ygrid, tgrid, data, params, bounds));
    return rcpp_result_gen;
END_RCPP
}
// full_sc_lhood
double full_sc_lhood(const NumericVector& xgrid, const NumericVector& ygrid, const NumericVector& tgrid, const NumericVector& tobs, const NumericMatrix& data, const NumericVector& params, const NumericVector& bounds);
RcppExport SEXP _ldmppr_full_sc_lhood(SEXP xgridSEXP, SEXP ygridSEXP, SEXP tgridSEXP, SEXP tobsSEXP, SEXP dataSEXP, SEXP paramsSEXP, SEXP boundsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type xgrid(xgridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type ygrid(ygridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type tgrid(tgridSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type tobs(tobsSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type bounds(boundsSEXP);
    rcpp_result_gen = Rcpp::wrap(full_sc_lhood(xgrid, ygrid, tgrid, tobs, data, params, bounds));
    return rcpp_result_gen;
END_RCPP
}
// spat_interaction
double spat_interaction(const NumericMatrix& Hist, const NumericVector& newp, const NumericVector& params);
RcppExport SEXP _ldmppr_spat_interaction(SEXP HistSEXP, SEXP newpSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type Hist(HistSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type newp(newpSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(spat_interaction(Hist, newp, params));
    return rcpp_result_gen;
END_RCPP
}
// interaction_st
NumericVector interaction_st(const NumericMatrix& data, const NumericVector& params);
RcppExport SEXP _ldmppr_interaction_st(SEXP dataSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(interaction_st(data, params));
    return rcpp_result_gen;
END_RCPP
}
// temporal_sc
double temporal_sc(const NumericVector& params, const double eval_t, const NumericVector& obs_t);
RcppExport SEXP _ldmppr_temporal_sc(SEXP paramsSEXP, SEXP eval_tSEXP, SEXP obs_tSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< const double >::type eval_t(eval_tSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type obs_t(obs_tSEXP);
    rcpp_result_gen = Rcpp::wrap(temporal_sc(params, eval_t, obs_t));
    return rcpp_result_gen;
END_RCPP
}
// sim_temporal_sc
NumericVector sim_temporal_sc(double Tmin, double Tmax, const NumericVector& params);
RcppExport SEXP _ldmppr_sim_temporal_sc(SEXP TminSEXP, SEXP TmaxSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type Tmin(TminSEXP);
    Rcpp::traits::input_parameter< double >::type Tmax(TmaxSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_temporal_sc(Tmin, Tmax, params));
    return rcpp_result_gen;
END_RCPP
}
// sim_spatial_sc
NumericMatrix sim_spatial_sc(const NumericVector& M_n, const NumericVector& params, int nsim_t, const NumericVector& xy_bounds);
RcppExport SEXP _ldmppr_sim_spatial_sc(SEXP M_nSEXP, SEXP paramsSEXP, SEXP nsim_tSEXP, SEXP xy_boundsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector& >::type M_n(M_nSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< int >::type nsim_t(nsim_tSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type xy_bounds(xy_boundsSEXP);
    rcpp_result_gen = Rcpp::wrap(sim_spatial_sc(M_n, params, nsim_t, xy_bounds));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ldmppr_vec_dist", (DL_FUNC) &_ldmppr_vec_dist, 2},
    {"_ldmppr_full_product", (DL_FUNC) &_ldmppr_full_product, 5},
    {"_ldmppr_C_theta2_i", (DL_FUNC) &_ldmppr_C_theta2_i, 6},
    {"_ldmppr_conditional_sum", (DL_FUNC) &_ldmppr_conditional_sum, 3},
    {"_ldmppr_conditional_sum_logical", (DL_FUNC) &_ldmppr_conditional_sum_logical, 3},
    {"_ldmppr_vec_to_mat_dist", (DL_FUNC) &_ldmppr_vec_to_mat_dist, 3},
    {"_ldmppr_dist_one_dim", (DL_FUNC) &_ldmppr_dist_one_dim, 2},
    {"_ldmppr_part_1_1_full", (DL_FUNC) &_ldmppr_part_1_1_full, 2},
    {"_ldmppr_part_1_2_full", (DL_FUNC) &_ldmppr_part_1_2_full, 2},
    {"_ldmppr_part_1_3_full", (DL_FUNC) &_ldmppr_part_1_3_full, 6},
    {"_ldmppr_part_1_4_full", (DL_FUNC) &_ldmppr_part_1_4_full, 2},
    {"_ldmppr_part_1_full", (DL_FUNC) &_ldmppr_part_1_full, 6},
    {"_ldmppr_part_2_full", (DL_FUNC) &_ldmppr_part_2_full, 6},
    {"_ldmppr_full_sc_lhood", (DL_FUNC) &_ldmppr_full_sc_lhood, 7},
    {"_ldmppr_spat_interaction", (DL_FUNC) &_ldmppr_spat_interaction, 3},
    {"_ldmppr_interaction_st", (DL_FUNC) &_ldmppr_interaction_st, 2},
    {"_ldmppr_temporal_sc", (DL_FUNC) &_ldmppr_temporal_sc, 3},
    {"_ldmppr_sim_temporal_sc", (DL_FUNC) &_ldmppr_sim_temporal_sc, 3},
    {"_ldmppr_sim_spatial_sc", (DL_FUNC) &_ldmppr_sim_spatial_sc, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_ldmppr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
