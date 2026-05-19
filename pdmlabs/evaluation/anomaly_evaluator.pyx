from libc.math cimport pow
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string

cdef enum overlap_cardinality:
    e_one
    e_reciprocal
    e_udf_gamma

cdef enum positional_bias:
    e_flat
    e_front
    e_middle
    e_back
    e_udf_delta

cdef enum e_metric:
    e_precision
    e_recall
    e_fscore

ctypedef int timestamp
ctypedef pair[timestamp, timestamp] time_range
ctypedef vector[time_range] time_intervals

cdef class Evaluator:
    cdef:
        double beta_
        double alpha_p_
        double alpha_r_
        overlap_cardinality gamma_p_
        overlap_cardinality gamma_r_
        positional_bias delta_p_
        positional_bias delta_r_
        double precision_
        double recall_
        double fscore_
        time_intervals real_anomalies_
        time_intervals predicted_anomalies_

    def __cinit__(self, time_intervals real, time_intervals predicted, double beta=1.0, double alpha_r=0.0,
                  overlap_cardinality gamma=e_one, positional_bias delta_p=e_flat, positional_bias delta_r=e_flat):
        self.beta_ = beta
        self.alpha_p_ = 0
        self.alpha_r_ = alpha_r
        self.gamma_p_ = gamma
        self.gamma_r_ = gamma
        self.delta_p_ = delta_p
        self.delta_r_ = delta_r
        self.precision_ = 0
        self.recall_ = 0
        self.fscore_ = 0
        self.real_anomalies_ = real
        self.predicted_anomalies_ = predicted

    cpdef void print_real_anomalies(self):
        print("Real Anomalies:")
        for i in self.real_anomalies_:
            print(f"[{i.first}, {i.second}]")

    cpdef void print_predicted_anomalies(self):
        print("Real Anomalies:")
        for i in self.predicted_anomalies_:
            print(f"[{i.first}, {i.second}]")

    cdef double udf_gamma_def(self, int overlap_count, e_metric m) nogil:
        cdef double return_val = 1.0
        return return_val

    cdef double udf_delta_def(self, timestamp t, timestamp anomaly_length, e_metric m) nogil:
        cdef double return_val = 1.0
        return return_val

    cdef double gamma_select(self, overlap_cardinality gamma, int overlap, e_metric m, string prec_or_recall) nogil:
        if gamma == e_one:
            return 1.0
        elif gamma == e_reciprocal:
            return 1.0 / overlap if overlap > 1 else 1.0
        elif gamma == e_udf_gamma:
            return 1.0 / self.udf_gamma_def(overlap, m) if overlap > 1 else 1.0
        return 1.0

    cdef double gamma_function(self, int overlap, e_metric m) nogil:
        cdef const char * metric_type
        if m == e_precision:
            metric_type = b"precision"
            return self.gamma_select(self.gamma_p_, overlap, m, metric_type)
        elif m == e_recall:
            metric_type = b"recall"
            return self.gamma_select(self.gamma_r_, overlap, m, metric_type)
        return 1.0

    cdef double delta_select(self, positional_bias delta, timestamp t, timestamp anomaly_length, e_metric m,
                             string prec_or_recall) nogil:

        if delta == e_flat:
            return 1.0
        elif delta == e_front:
            return anomaly_length - t + 1
        elif delta == e_middle:
            return t if t <= anomaly_length / 2 else anomaly_length - t + 1
        elif delta == e_back:
            return t
        elif delta == e_udf_delta:
            return self.udf_delta_def(t, anomaly_length, m)
        return 1.0

    cdef double delta_function(self, timestamp t, timestamp anomaly_length, e_metric m) nogil:
        cdef const char * metric_type  # Declare a C string

        if m == e_precision:
            metric_type = b"precision"  # Use byte string (C string)
            return self.delta_select(self.delta_p_, t, anomaly_length, m, metric_type)
        elif m == e_recall:
            metric_type = b"recall"
            return self.delta_select(self.delta_r_, t, anomaly_length, m, metric_type)

    cdef double omega_function(self, time_range range, time_range overlap, e_metric m) nogil:
        cdef:
            timestamp anomaly_length = range.second - range.first + 1
            double my_positional_bias = 0, max_positional_bias = 0, temp_bias = 0
            timestamp i, j
        for i from 1 <= i < anomaly_length + 1:
            temp_bias = self.delta_function(i, anomaly_length, m)
            max_positional_bias += temp_bias

            j = range.first + i - 1
            if overlap.first <= j <= overlap.second:
                my_positional_bias += temp_bias
        return my_positional_bias / max_positional_bias if max_positional_bias > 0 else 0

    cdef double compute_omega_reward(self, time_range r1, time_range r2, int& overlap_count, e_metric m) nogil:
        if r1.second < r2.first or r1.first > r2.second:
            return 0
        overlap_count += 1
        cdef time_range overlap
        overlap.first = r1.first if r1.first > r2.first else r2.first
        overlap.second = r1.second if r1.second < r2.second else r2.second
        return self.omega_function(r1, overlap, m)

    cpdef double compute_precision(self):
        cdef:
            time_range range_p, range_r
            double existence_reward, omega_reward, overlap_reward
            int overlap_count
            double precision = 0.0

        if len(self.predicted_anomalies_) == 0:
            return 0.0

        for i in self.predicted_anomalies_:
            range_p = i
            omega_reward = 0
            overlap_count = 0

            for j in self.real_anomalies_:
                range_r = j
                omega_reward += self.compute_omega_reward(range_p, range_r, overlap_count, e_precision)
            overlap_reward = self.gamma_function(overlap_count, e_precision) * omega_reward
            existence_reward = 1 if overlap_count > 0 else 0
            precision += self.alpha_p_ * existence_reward + (1.0 - self.alpha_p_) * overlap_reward
        self.precision_ = precision / len(self.predicted_anomalies_)
        return precision / len(self.predicted_anomalies_)

    cpdef double compute_recall(self):
        cdef:
            time_range range_p, range_r
            double existence_reward, omega_reward, overlap_reward
            int overlap_count
            double recall = 0.0

        if len(self.real_anomalies_)==0:
            return 0.0

        for i in self.real_anomalies_:
            range_r = i
            omega_reward = 0
            overlap_count = 0

            for j in self.predicted_anomalies_:
                range_p = j
                omega_reward += self.compute_omega_reward(range_r, range_p, overlap_count, e_recall)
            overlap_reward = self.gamma_function(overlap_count, e_recall) * omega_reward
            existence_reward = 1 if overlap_count > 0 else 0
            recall += self.alpha_r_ * existence_reward + (1.0 - self.alpha_r_) * overlap_reward
        self.recall_ = recall / len(self.real_anomalies_)
        return self.recall_

    cpdef double compute_fscore(self):
        cdef double beta_sqr = pow(self.beta_, 2.0)
        return (1 + beta_sqr) * (self.precision_ * self.recall_) / (
                    beta_sqr * self.precision_ + self.recall_) if self.precision_ + self.recall_ > 0 else 0




cdef vector[pair[int, int]] read_file(str filename, int* count):
    cdef vector[pair[int, int]] anomalies
    cdef int i = 0
    cdef int label
    cdef pair[int, int] range
    cdef bint range_started = False

    with open(filename, 'r') as f:
        for line in f:
            label = int(line.strip())
            if label == 1:  # Anomaly
                if not range_started:
                    range_started = True
                    range.first = i
                    range.second = i
                else:
                    range.second = i
            elif label == 0:  # Not anomaly
                if range_started:
                    anomalies.push_back(range)
                    range_started = False
            else:
                raise ValueError("Invalid anomaly label!")
            i += 1

    if range_started:  # Last read label was an anomaly
        anomalies.push_back(range)

    count[0] = i
    return anomalies

def evaluate(str real_filename, str predicted_filename, double beta=1.0, double alpha_r=0.5, str cardinality="one", str bias_p="flat", str bias_r="flat"):
    cdef int real_count = 0, predicted_count = 0
    cdef vector[pair[int, int]] real_anomalies, predicted_anomalies

    real_anomalies = read_file(real_filename, &real_count)
    predicted_anomalies = read_file(predicted_filename, &predicted_count)

    if real_count != predicted_count:
        raise ValueError("Number of data items are different!")
    if real_count == 0:
        raise ValueError("No data items!")

    cdef overlap_cardinality gamma
    cdef positional_bias delta_p, delta_r

    if cardinality == "one":
        gamma = e_one
    elif cardinality == "reciprocal":
        gamma = e_reciprocal
    elif cardinality == "udf_gamma":
        gamma = e_udf_gamma
    else:
        raise ValueError("Invalid overlap cardinality value!")

    def convert_bias(str bias):
        if bias == "flat":
            return e_flat
        elif bias == "front":
            return e_front
        elif bias == "middle":
            return e_middle
        elif bias == "back":
            return e_back
        elif bias == "udf_delta":
            return e_udf_delta
        else:
            raise ValueError("Invalid positional bias value!")

    delta_p = convert_bias(bias_p)
    delta_r = convert_bias(bias_r)
    cdef Evaluator e = Evaluator(real_anomalies, predicted_anomalies, beta, alpha_r, gamma, delta_p, delta_r)


    return {
        "Precision": e.compute_precision(),
        "Recall": e.compute_recall(),
        "F-Score": e.compute_fscore()
    }