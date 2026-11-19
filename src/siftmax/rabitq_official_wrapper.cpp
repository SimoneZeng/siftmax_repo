#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <cmath>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/space.hpp"

namespace {

int parse_metric(PyObject* metric_obj) {
    const char* metric = "ip";
    if (metric_obj != nullptr && metric_obj != Py_None) {
        metric = PyUnicode_AsUTF8(metric_obj);
        if (metric == nullptr) {
            return -1;
        }
    }
    if (std::strcmp(metric, "ip") == 0 || std::strcmp(metric, "IP") == 0) {
        return rabitqlib::METRIC_IP;
    }
    if (std::strcmp(metric, "l2") == 0 || std::strcmp(metric, "L2") == 0) {
        return rabitqlib::METRIC_L2;
    }
    PyErr_SetString(PyExc_ValueError, "metric must be 'ip' or 'l2'");
    return -1;
}

bool parse_bool(PyObject* obj, bool default_value) {
    if (obj == nullptr || obj == Py_None) {
        return default_value;
    }
    int value = PyObject_IsTrue(obj);
    return value > 0;
}

inline size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

// Maintains a monotonic shared top-k threshold for parallel pruning.
class LockFreeTopKScores {
public:
    explicit LockFreeTopKScores(size_t top_k)
        : top_k_(top_k),
          slots_(top_k > 1 ? new std::atomic<float>[top_k] : nullptr),
          tau_(-std::numeric_limits<float>::infinity()) {
        for (size_t i = 0; i < top_k_; ++i) {
            if (top_k_ > 1) {
                slots_[i].store(
                    -std::numeric_limits<float>::infinity(),
                    std::memory_order_relaxed);
            }
        }
    }

    float tau() const {
        return tau_.load(std::memory_order_relaxed);
    }

    void insert(float score) {
        if (top_k_ == 0 || std::isnan(score)) {
            return;
        }
        if (top_k_ == 1) {
            raise_tau(score);
            return;
        }
        if (score <= tau()) {
            return;
        }

        while (true) {
            size_t min_index = 0;
            float min_score = slots_[0].load(std::memory_order_relaxed);
            for (size_t i = 1; i < top_k_; ++i) {
                const float slot_score = slots_[i].load(std::memory_order_relaxed);
                if (slot_score < min_score) {
                    min_score = slot_score;
                    min_index = i;
                }
            }
            if (score <= min_score) {
                return;
            }

            if (slots_[min_index].compare_exchange_weak(
                    min_score,
                    score,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                publish_tau();
                return;
            }
        }
    }

private:
    void publish_tau() {
        float min_score = slots_[0].load(std::memory_order_relaxed);
        for (size_t i = 1; i < top_k_; ++i) {
            min_score = std::min(
                min_score,
                slots_[i].load(std::memory_order_relaxed));
        }
        raise_tau(min_score);
    }

    void raise_tau(float candidate) {
        float current = tau_.load(std::memory_order_relaxed);
        while (candidate > current &&
               !tau_.compare_exchange_weak(
                   current,
                   candidate,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }

    size_t top_k_;
    std::unique_ptr<std::atomic<float>[]> slots_;
    std::atomic<float> tau_;
};

static_assert(
    std::atomic<float>::is_always_lock_free,
    "one-bit lock-free top-k requires lock-free float atomics");

// Computes FastScan distance estimates without materializing lower bounds.
inline void split_batch_estdist_no_low(
    const char* batch_data,
    const rabitqlib::SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    constexpr size_t kSafeChunkDim = 1024;
    rabitqlib::ConstBatchDataMap<float> cur_batch(batch_data, padded_dim);
    alignas(64) std::array<int32_t, rabitqlib::fastscan::kBatchSize> accu_arr;
    const auto* codes_ptr = cur_batch.bin_code();
    const auto* lut_ptr = q_obj.lut();

    if (use_hacc) {
        if (padded_dim <= kSafeChunkDim) {
            rabitqlib::fastscan::accumulate_hacc(
                codes_ptr,
                lut_ptr,
                accu_arr.data(),
                padded_dim);
        } else {
            accu_arr.fill(0);
            std::array<int32_t, rabitqlib::fastscan::kBatchSize> accu_res;
            size_t remaining_dim = padded_dim;
            while (remaining_dim > kSafeChunkDim) {
                rabitqlib::fastscan::accumulate_hacc(
                    codes_ptr,
                    lut_ptr,
                    accu_res.data(),
                    kSafeChunkDim);
                codes_ptr += kSafeChunkDim << 2;
                lut_ptr += kSafeChunkDim << 3;
                for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                    accu_arr[i] += accu_res[i];
                }
                remaining_dim -= kSafeChunkDim;
            }
            rabitqlib::fastscan::accumulate_hacc(
                codes_ptr,
                lut_ptr,
                accu_res.data(),
                remaining_dim);
            for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                accu_arr[i] += accu_res[i];
            }
        }
    } else {
        std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu_res;
        if (padded_dim <= kSafeChunkDim) {
            rabitqlib::fastscan::accumulate(
                codes_ptr,
                lut_ptr,
                accu_res.data(),
                padded_dim);
            for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                accu_arr[i] = static_cast<int32_t>(accu_res[i]);
            }
        } else {
            accu_arr.fill(0);
            size_t remaining_dim = padded_dim;
            while (remaining_dim > kSafeChunkDim) {
                rabitqlib::fastscan::accumulate(
                    codes_ptr,
                    lut_ptr,
                    accu_res.data(),
                    kSafeChunkDim);
                codes_ptr += kSafeChunkDim << 2;
                lut_ptr += kSafeChunkDim << 2;
                for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                    accu_arr[i] += static_cast<int32_t>(accu_res[i]);
                }
                remaining_dim -= kSafeChunkDim;
            }
            rabitqlib::fastscan::accumulate(
                codes_ptr,
                lut_ptr,
                accu_res.data(),
                remaining_dim);
            for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                accu_arr[i] += static_cast<int32_t>(accu_res[i]);
            }
        }
    }

    const float delta = q_obj.delta();
    const float sum_vl_lut = q_obj.sum_vl_lut();
    const float g_add = q_obj.g_add();
    const float k1xsumq = q_obj.k1xsumq();
    const float* f_add = cur_batch.f_add();
    const float* f_rescale = cur_batch.f_rescale();
#pragma omp simd
    for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
        const float ip_x0 = delta * static_cast<float>(accu_arr[i]) + sum_vl_lut;
        ip_x0_qr[i] = ip_x0;
        est_distance[i] = f_add[i] + g_add + f_rescale[i] * (ip_x0 + k1xsumq);
    }
}

inline void split_batch_estdist_no_low(
    const char* batch_data,
    const rabitqlib::SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    bool use_hacc
) {
    std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr_scratch;
    split_batch_estdist_no_low(
        batch_data,
        q_obj,
        padded_dim,
        est_distance,
        ip_x0_qr_scratch.data(),
        use_hacc);
}

struct ArrayRef {
    PyArrayObject* ptr = nullptr;

    ArrayRef() = default;
    explicit ArrayRef(PyArrayObject* array) : ptr(array) {}
    ~ArrayRef() { Py_XDECREF(ptr); }

    ArrayRef(const ArrayRef&) = delete;
    ArrayRef& operator=(const ArrayRef&) = delete;

    ArrayRef(ArrayRef&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    ArrayRef& operator=(ArrayRef&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    [[nodiscard]] PyArrayObject* get() const { return ptr; }
    [[nodiscard]] explicit operator bool() const { return ptr != nullptr; }

    [[nodiscard]] PyObject* release_object() {
        PyObject* obj = reinterpret_cast<PyObject*>(ptr);
        ptr = nullptr;
        return obj;
    }
};

// Encodes rotated document tokens into packed one-bit and extra-bit layouts.
PyObject* encode_pack_rabitq(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* data_obj = nullptr;
    PyObject* centroids_obj = nullptr;
    PyObject* centroid_ids_obj = nullptr;
    PyObject* doc_offsets_obj = nullptr;
    PyObject* metric_obj = nullptr;
    int total_bits = 1;
    int dim = 0;
    int encode_batch_size = 200000;
    static const char* kwlist[] = {
        "rotated_vectors",
        "centroids_rot",
        "centroid_ids_compact",
        "doc_token_offsets",
        "total_bits",
        "dim",
        "metric",
        "encode_batch_size",
        nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOii|Oi",
            const_cast<char**>(kwlist),
            &data_obj,
            &centroids_obj,
            &centroid_ids_obj,
            &doc_offsets_obj,
            &total_bits,
            &dim,
            &metric_obj,
            &encode_batch_size)) {
        return nullptr;
    }

    const int metric_type = parse_metric(metric_obj);
    if (metric_type < 0) {
        return nullptr;
    }
    if (dim <= 1 || dim % 64 != 0) {
        PyErr_SetString(PyExc_ValueError, "dim must be > 1 and divisible by 64 for fused RaBitQ packing");
        return nullptr;
    }
    if (total_bits < 1 || total_bits > 8) {
        PyErr_SetString(PyExc_ValueError, "total_bits must be in [1, 8] for encode_pack_rabitq");
        return nullptr;
    }
    if (encode_batch_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "encode_batch_size must be positive");
        return nullptr;
    }
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(data_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    PyArrayObject* centroids_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(centroids_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    PyArrayObject* centroid_ids_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(centroid_ids_obj, NPY_UINT16, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    PyArrayObject* doc_offsets_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(doc_offsets_obj, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (data_arr == nullptr || centroids_arr == nullptr || centroid_ids_arr == nullptr ||
        doc_offsets_arr == nullptr) {
        Py_XDECREF(data_arr);
        Py_XDECREF(centroids_arr);
        Py_XDECREF(centroid_ids_arr);
        Py_XDECREF(doc_offsets_arr);
        return nullptr;
    }

    const npy_intp n_tokens = PyArray_DIM(data_arr, 0);
    const npy_intp n_centroids = PyArray_DIM(centroids_arr, 0);
    const npy_intp n_docs = PyArray_DIM(doc_offsets_arr, 0) - 1;
    if (PyArray_NDIM(data_arr) != 2 || PyArray_DIM(data_arr, 1) != dim ||
        PyArray_NDIM(centroids_arr) != 2 || PyArray_DIM(centroids_arr, 1) != dim ||
        n_centroids <= 0 ||
        PyArray_NDIM(centroid_ids_arr) != 1 ||
        PyArray_NDIM(doc_offsets_arr) != 1 ||
        n_docs < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid encode_pack_rabitq input shapes");
        Py_DECREF(data_arr);
        Py_DECREF(centroids_arr);
        Py_DECREF(centroid_ids_arr);
        Py_DECREF(doc_offsets_arr);
        return nullptr;
    }
    if (PyArray_DIM(centroid_ids_arr, 0) != n_tokens) {
        PyErr_SetString(PyExc_ValueError, "centroid_ids_compact length must match token count for IVF centers");
        Py_DECREF(data_arr);
        Py_DECREF(centroids_arr);
        Py_DECREF(centroid_ids_arr);
        Py_DECREF(doc_offsets_arr);
        return nullptr;
    }
    if (n_centroids > static_cast<npy_intp>(std::numeric_limits<uint16_t>::max()) + 1) {
        PyErr_SetString(PyExc_ValueError, "uint16 centroid id layout supports at most 65536 IVF centroids");
        Py_DECREF(data_arr);
        Py_DECREF(centroids_arr);
        Py_DECREF(centroid_ids_arr);
        Py_DECREF(doc_offsets_arr);
        return nullptr;
    }
    const auto* centroid_ids = static_cast<const uint16_t*>(PyArray_DATA(centroid_ids_arr));
    for (npy_intp i = 0; i < n_tokens; ++i) {
        if (static_cast<npy_intp>(centroid_ids[i]) >= n_centroids) {
            PyErr_SetString(PyExc_ValueError, "centroid_ids_compact contains an out-of-range centroid id");
            Py_DECREF(data_arr);
            Py_DECREF(centroids_arr);
            Py_DECREF(centroid_ids_arr);
            Py_DECREF(doc_offsets_arr);
            return nullptr;
        }
    }

    const auto* doc_offsets = static_cast<const int64_t*>(PyArray_DATA(doc_offsets_arr));
    size_t total_batches = 0;
    for (npy_intp d = 0; d < n_docs; ++d) {
        if (doc_offsets[d] < 0 || doc_offsets[d + 1] < doc_offsets[d] ||
            doc_offsets[d + 1] > n_tokens) {
            PyErr_SetString(PyExc_ValueError, "doc_token_offsets must be monotonic and within token count");
            Py_DECREF(data_arr);
            Py_DECREF(centroids_arr);
            Py_DECREF(centroid_ids_arr);
            Py_DECREF(doc_offsets_arr);
            return nullptr;
        }
        total_batches += ceil_div(static_cast<size_t>(doc_offsets[d + 1] - doc_offsets[d]),
                                  rabitqlib::fastscan::kBatchSize);
    }

    const size_t code_size = static_cast<size_t>(dim) / 8;
    const size_t batch_bytes = rabitqlib::BatchDataMap<float>::data_bytes(static_cast<size_t>(dim));
    npy_intp data_dims[1] = {static_cast<npy_intp>(total_batches * batch_bytes)};
    npy_intp batch_offset_dims[1] = {n_docs + 1};
    PyArrayObject* batch_data_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, data_dims, NPY_UINT8, 0));
    PyArrayObject* doc_batch_offsets_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, batch_offset_dims, NPY_INT64));
    if (batch_data_arr == nullptr || doc_batch_offsets_arr == nullptr) {
        Py_XDECREF(batch_data_arr);
        Py_XDECREF(doc_batch_offsets_arr);
        Py_DECREF(data_arr);
        Py_DECREF(centroids_arr);
        Py_DECREF(centroid_ids_arr);
        Py_DECREF(doc_offsets_arr);
        return nullptr;
    }

    auto* doc_batch_offsets = static_cast<int64_t*>(PyArray_DATA(doc_batch_offsets_arr));
    doc_batch_offsets[0] = 0;
    size_t batch_id = 0;
    for (npy_intp d = 0; d < n_docs; ++d) {
        batch_id += ceil_div(static_cast<size_t>(doc_offsets[d + 1] - doc_offsets[d]),
                             rabitqlib::fastscan::kBatchSize);
        doc_batch_offsets[d + 1] = static_cast<int64_t>(batch_id);
    }

    const size_t ex_bits = total_bits > 1 ? static_cast<size_t>(total_bits - 1) : 0;
    const size_t ex_stride = ex_bits > 0
        ? rabitqlib::ExDataMap<float>::data_bytes(static_cast<size_t>(dim), ex_bits)
        : 0;
    npy_intp ex_dims[1] = {static_cast<npy_intp>(static_cast<size_t>(n_tokens) * ex_stride)};
    PyArrayObject* ex_data_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, ex_dims, NPY_UINT8, 0));
    if (ex_data_arr == nullptr) {
        Py_DECREF(batch_data_arr);
        Py_DECREF(doc_batch_offsets_arr);
        Py_DECREF(data_arr);
        Py_DECREF(centroids_arr);
        Py_DECREF(centroid_ids_arr);
        Py_DECREF(doc_offsets_arr);
        return nullptr;
    }

    const auto* data = static_cast<const float*>(PyArray_DATA(data_arr));
    const auto* centroids = static_cast<const float*>(PyArray_DATA(centroids_arr));
    auto* batch_data = static_cast<char*>(PyArray_DATA(batch_data_arr));
    auto* ex_data = static_cast<char*>(PyArray_DATA(ex_data_arr));
    rabitqlib::quant::RabitqConfig config =
        rabitqlib::quant::faster_config(static_cast<size_t>(dim), static_cast<size_t>(std::max(total_bits, 2)));
    const uint8_t ex_mask = ex_bits > 0
        ? static_cast<uint8_t>((static_cast<unsigned int>(1U) << ex_bits) - 1U)
        : static_cast<uint8_t>(0);

    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic)
    for (npy_intp d = 0; d < n_docs; ++d) {
        std::vector<uint8_t> batch_codes(rabitqlib::fastscan::kBatchSize * code_size, 0);
        std::vector<uint8_t> full_code(static_cast<size_t>(dim), 0);
        std::vector<uint8_t> extra_code(static_cast<size_t>(dim), 0);
        std::array<float, rabitqlib::fastscan::kBatchSize> f_add{};
        std::array<float, rabitqlib::fastscan::kBatchSize> f_rescale{};
        std::array<float, rabitqlib::fastscan::kBatchSize> f_error{};

        const size_t begin = static_cast<size_t>(doc_offsets[d]);
        const size_t end = static_cast<size_t>(doc_offsets[d + 1]);
        size_t local_batch_id = static_cast<size_t>(doc_batch_offsets[d]);
        for (size_t token = begin; token < end; token += rabitqlib::fastscan::kBatchSize) {
            const size_t count = std::min(rabitqlib::fastscan::kBatchSize, end - token);
            std::fill(batch_codes.begin(), batch_codes.end(), 0);
            f_add.fill(0.0F);
            f_rescale.fill(0.0F);
            f_error.fill(0.0F);

            for (size_t i = 0; i < count; ++i) {
                const size_t token_id = token + i;
                const uint16_t centroid_id = centroid_ids[token_id];
                const float* centroid = centroids +
                    (static_cast<size_t>(centroid_id) * static_cast<size_t>(dim));

                rabitqlib::quant::quantize_compact_one_bit<float, uint8_t*>(
                    data + (token_id * static_cast<size_t>(dim)),
                    centroid,
                    static_cast<size_t>(dim),
                    batch_codes.data() + (i * code_size),
                    f_add[i],
                    f_rescale[i],
                    f_error[i],
                    static_cast<rabitqlib::MetricType>(metric_type));

                if (!std::isfinite(f_error[i])) {
                    bool zero_residual = true;
                    const float* token_data =
                        data + (token_id * static_cast<size_t>(dim));
                    for (int j = 0; j < dim; ++j) {
                        if (token_data[j] != centroid[j]) {
                            zero_residual = false;
                            break;
                        }
                    }
                    if (zero_residual) {
                        f_error[i] = 0.0F;
                    }
                }

                if (ex_bits > 0) {
                    float f_add_ex = 0.0F;
                    float f_rescale_ex = 0.0F;
                    float f_error_ex = 0.0F;
                    rabitqlib::quant::quantize_full_single<float, uint8_t>(
                        data + (token_id * static_cast<size_t>(dim)),
                        centroid,
                        static_cast<size_t>(dim),
                        static_cast<size_t>(total_bits),
                        full_code.data(),
                        f_add_ex,
                        f_rescale_ex,
                        f_error_ex,
                        static_cast<rabitqlib::MetricType>(metric_type),
                        config);
                    for (size_t j = 0; j < static_cast<size_t>(dim); ++j) {
                        extra_code[j] = full_code[j] & ex_mask;
                    }
                    char* ex_ptr = ex_data + (token_id * ex_stride);
                    rabitqlib::ExDataMap<float> cur_ex(
                        ex_ptr, static_cast<size_t>(dim), ex_bits);
                    rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                        extra_code.data(),
                        cur_ex.ex_code(),
                        static_cast<size_t>(dim),
                        ex_bits);
                    cur_ex.f_add_ex() = f_add_ex;
                    cur_ex.f_rescale_ex() = f_rescale_ex;
                }
            }

            char* batch_ptr = batch_data + (local_batch_id * batch_bytes);
            rabitqlib::BatchDataMap<float> cur_batch(batch_ptr, static_cast<size_t>(dim));
            rabitqlib::fastscan::pack_codes(
                static_cast<size_t>(dim),
                batch_codes.data(),
                rabitqlib::fastscan::kBatchSize,
                cur_batch.bin_code());
            for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                cur_batch.f_add()[i] = f_add[i];
                cur_batch.f_rescale()[i] = f_rescale[i];
                cur_batch.f_error()[i] = f_error[i];
            }
            ++local_batch_id;
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(data_arr);
    Py_DECREF(centroids_arr);
    Py_DECREF(centroid_ids_arr);
    Py_DECREF(doc_offsets_arr);

    return Py_BuildValue(
        "NNN",
        reinterpret_cast<PyObject*>(batch_data_arr),
        reinterpret_cast<PyObject*>(doc_batch_offsets_arr),
        reinterpret_cast<PyObject*>(ex_data_arr));
}

// Reranks candidates with one-bit bounds, progressive n-bit refinement, and top-k selection.
PyObject* rerank_topk_rabitq(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* q_obj = nullptr;
    PyObject* candidate_obj = nullptr;
    PyObject* onebit_obj = nullptr;
    PyObject* ex_data_obj = nullptr;
    PyObject* doc_offsets_obj = nullptr;
    PyObject* doc_batch_offsets_obj = nullptr;
    PyObject* centroids_obj = nullptr;
    PyObject* centroid_norm_sq_obj = nullptr;
    PyObject* centroid_ids_obj = nullptr;
    PyObject* doc_hit_centroids_obj = nullptr;
    PyObject* doc_centroid_offsets_obj = nullptr;
    PyObject* use_hacc_obj = nullptr;
    int total_bits = 1;
    int dim = 0;
    int top_k = 0;
    double alpha_geometry = 1.0;
    static const char* kwlist[] = {
        "q_rot",
        "candidate_doc_ids",
        "onebit_batch_data",
        "ex_data",
        "doc_token_offsets",
        "doc_batch_offsets",
        "centroids_rot",
        "centroid_norm_sq",
        "centroid_ids",
        "doc_hit_centroids",
        "doc_centroid_offsets",
        "total_bits",
        "dim",
        "top_k",
        "alpha_geometry",
        "use_hacc",
        nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOOOOOiiid|O",
            const_cast<char**>(kwlist),
            &q_obj,
            &candidate_obj,
            &onebit_obj,
            &ex_data_obj,
            &doc_offsets_obj,
            &doc_batch_offsets_obj,
            &centroids_obj,
            &centroid_norm_sq_obj,
            &centroid_ids_obj,
            &doc_hit_centroids_obj,
            &doc_centroid_offsets_obj,
            &total_bits,
            &dim,
            &top_k,
            &alpha_geometry,
            &use_hacc_obj)) {
        return nullptr;
    }

    const size_t ex_bits = total_bits > 1 ? static_cast<size_t>(total_bits - 1) : 0;
    const bool use_hacc = parse_bool(use_hacc_obj, true);

    ArrayRef q_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(q_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef candidate_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(candidate_obj, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef onebit_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(onebit_obj, NPY_UINT8, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef ex_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(ex_data_obj, NPY_UINT8, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef doc_offsets_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(doc_offsets_obj, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef doc_batch_offsets_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(doc_batch_offsets_obj, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef centroids_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(centroids_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef centroid_norm_sq_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(centroid_norm_sq_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef centroid_ids_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(centroid_ids_obj, NPY_UINT16, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef doc_hit_centroids_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(doc_hit_centroids_obj, NPY_UINT16, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    ArrayRef doc_centroid_offsets_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(doc_centroid_offsets_obj, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)));
    if (!q_arr || !candidate_arr || !onebit_arr || !ex_arr || !doc_offsets_arr ||
        !doc_batch_offsets_arr || !centroids_arr || !centroid_norm_sq_arr || !centroid_ids_arr ||
        !doc_hit_centroids_arr || !doc_centroid_offsets_arr) {
        return nullptr;
    }

    const npy_intp q_len = PyArray_DIM(q_arr.get(), 0);
    const npy_intp n_cands = PyArray_DIM(candidate_arr.get(), 0);
    const npy_intp n_docs = PyArray_DIM(doc_offsets_arr.get(), 0) - 1;
    if (top_k <= 0) {
        PyErr_SetString(PyExc_ValueError, "top_k must be positive");
        return nullptr;
    }
    if (n_cands < static_cast<npy_intp>(top_k)) {
        PyErr_SetString(PyExc_ValueError, "number of candidates must be greater than or equal to top_k");
        return nullptr;
    }
    if (PyArray_NDIM(q_arr.get()) != 2 || PyArray_DIM(q_arr.get(), 1) != dim ||
        PyArray_NDIM(candidate_arr.get()) != 1 || PyArray_NDIM(onebit_arr.get()) != 1 ||
        PyArray_NDIM(ex_arr.get()) != 1 || PyArray_NDIM(doc_offsets_arr.get()) != 1 ||
        PyArray_NDIM(doc_batch_offsets_arr.get()) != 1 || PyArray_DIM(doc_batch_offsets_arr.get(), 0) != n_docs + 1 ||
        PyArray_NDIM(centroids_arr.get()) != 2 || PyArray_DIM(centroids_arr.get(), 1) != dim ||
        PyArray_DIM(centroids_arr.get(), 0) <= 0 ||
        PyArray_NDIM(centroid_norm_sq_arr.get()) != 1 ||
        PyArray_DIM(centroid_norm_sq_arr.get(), 0) != PyArray_DIM(centroids_arr.get(), 0) ||
        PyArray_NDIM(centroid_ids_arr.get()) != 1 ||
        PyArray_NDIM(doc_hit_centroids_arr.get()) != 1 ||
        PyArray_NDIM(doc_centroid_offsets_arr.get()) != 1 ||
        n_docs < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid rerank_topk_rabitq input shapes");
        return nullptr;
    }
    const npy_intp n_centroids = PyArray_DIM(centroids_arr.get(), 0);
    if (n_centroids > static_cast<npy_intp>(std::numeric_limits<uint16_t>::max()) + 1) {
        PyErr_SetString(PyExc_ValueError, "uint16 centroid id layout supports at most 65536 IVF centroids");
        return nullptr;
    }
    if (PyArray_DIM(doc_centroid_offsets_arr.get(), 0) != n_docs + 1 ||
        PyArray_DIM(doc_hit_centroids_arr.get(), 0) < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid per-document centroid CSR shapes");
        return nullptr;
    }
    if (!std::isfinite(alpha_geometry) || alpha_geometry < 0.0 || alpha_geometry > 1.0) {
        PyErr_SetString(PyExc_ValueError, "alpha_geometry must be finite and in [0, 1]");
        return nullptr;
    }

    const auto* q = static_cast<const float*>(PyArray_DATA(q_arr.get()));
    const auto* candidates = static_cast<const int64_t*>(PyArray_DATA(candidate_arr.get()));
    const auto* onebit_data = static_cast<const char*>(PyArray_DATA(onebit_arr.get()));
    const auto* ex_data = static_cast<const char*>(PyArray_DATA(ex_arr.get()));
    const auto* doc_offsets = static_cast<const int64_t*>(PyArray_DATA(doc_offsets_arr.get()));
    const auto* doc_batch_offsets = static_cast<const int64_t*>(PyArray_DATA(doc_batch_offsets_arr.get()));
    const auto* centroids = static_cast<const float*>(PyArray_DATA(centroids_arr.get()));
    const auto* centroid_norm_sq = static_cast<const float*>(PyArray_DATA(centroid_norm_sq_arr.get()));
    const auto* centroid_ids = static_cast<const uint16_t*>(PyArray_DATA(centroid_ids_arr.get()));
    const auto* doc_hit_centroids = static_cast<const uint16_t*>(PyArray_DATA(doc_hit_centroids_arr.get()));
    const auto* doc_centroid_offsets = static_cast<const int64_t*>(PyArray_DATA(doc_centroid_offsets_arr.get()));

    const size_t batch_bytes = rabitqlib::BatchDataMap<float>::data_bytes(static_cast<size_t>(dim));
    const size_t ex_stride = ex_bits > 0
        ? rabitqlib::ExDataMap<float>::data_bytes(static_cast<size_t>(dim), ex_bits)
        : 0;
    rabitqlib::ex_ipfunc ip_func = ex_bits > 0 ? rabitqlib::select_excode_ipfunc(ex_bits) : nullptr;

    using Clock = std::chrono::steady_clock;

    const auto query_lut_build_start = Clock::now();
    std::deque<rabitqlib::SplitBatchQuery<float>> split_queries; 
    for (npy_intp qi = 0; qi < q_len; ++qi) {
        const float* q_ptr = q + (static_cast<size_t>(qi) * static_cast<size_t>(dim));
        split_queries.emplace_back(
            q_ptr,
            static_cast<size_t>(dim),
            ex_bits,
            rabitqlib::METRIC_IP,
            use_hacc);
        split_queries.back().set_g_add(0.0F, 0.0F);
    }
    const auto query_lut_build_end = Clock::now();

    const auto query_sim_precompute_start = Clock::now();
    std::vector<double> query_cosines(
        static_cast<size_t>(q_len) * static_cast<size_t>(q_len), 0.0);
    std::vector<double> query_orthogonal_sines(
        static_cast<size_t>(q_len) * static_cast<size_t>(q_len), 0.0);
    std::vector<int32_t> onebit_query_order(static_cast<size_t>(q_len));
    for (npy_intp qi = 0; qi < q_len; ++qi) {
        onebit_query_order[static_cast<size_t>(qi)] = static_cast<int32_t>(qi);
    }
    if (q_len > 1) {
        for (npy_intp qi = 0; qi < q_len; ++qi) {
            query_cosines[static_cast<size_t>(qi) * static_cast<size_t>(q_len) + static_cast<size_t>(qi)] = 1.0;
            const float* qi_ptr = q + static_cast<size_t>(qi) * static_cast<size_t>(dim);
            for (npy_intp qj = qi + 1; qj < q_len; ++qj) {
                const float* qj_ptr = q + static_cast<size_t>(qj) * static_cast<size_t>(dim);
                double dot = 0.0;
                for (int d = 0; d < dim; ++d) {
                    dot += static_cast<double>(qi_ptr[d]) * qj_ptr[d];
                }
                const double cosine = std::max(-1.0, std::min(1.0, dot));
                query_cosines[static_cast<size_t>(qi) * static_cast<size_t>(q_len) + static_cast<size_t>(qj)] = cosine;
                query_cosines[static_cast<size_t>(qj) * static_cast<size_t>(q_len) + static_cast<size_t>(qi)] = cosine;
                const double orthogonal_sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
                query_orthogonal_sines[static_cast<size_t>(qi) * static_cast<size_t>(q_len) + static_cast<size_t>(qj)] = orthogonal_sine;
                query_orthogonal_sines[static_cast<size_t>(qj) * static_cast<size_t>(q_len) + static_cast<size_t>(qi)] = orthogonal_sine;
            }
        }
        std::vector<uint8_t> selected(static_cast<size_t>(q_len), 0);
        int32_t first = 0;
        double lowest_mean = std::numeric_limits<double>::infinity();
        for (npy_intp qi = 0; qi < q_len; ++qi) {
            double mean = 0.0;
            for (npy_intp qj = 0; qj < q_len; ++qj) {
                if (qi != qj) {
                    mean += query_cosines[static_cast<size_t>(qi) * static_cast<size_t>(q_len) + static_cast<size_t>(qj)];
                }
            }
            if (mean < lowest_mean) {
                lowest_mean = mean;
                first = static_cast<int32_t>(qi);
            }
        }
        onebit_query_order[0] = first;
        selected[static_cast<size_t>(first)] = 1;
        std::vector<double> max_similarity_to_selected(
            static_cast<size_t>(q_len),
            -std::numeric_limits<double>::infinity());
        for (npy_intp qi = 0; qi < q_len; ++qi) {
            if (selected[static_cast<size_t>(qi)] == 0) {
                max_similarity_to_selected[static_cast<size_t>(qi)] =
                    query_cosines[
                        static_cast<size_t>(qi) * static_cast<size_t>(q_len) +
                        static_cast<size_t>(first)];
            }
        }
        for (npy_intp pos = 1; pos < q_len; ++pos) {
            int32_t best = -1;
            double best_max_similarity = std::numeric_limits<double>::infinity();
            for (npy_intp qi = 0; qi < q_len; ++qi) {
                if (selected[static_cast<size_t>(qi)] != 0) {
                    continue;
                }
                const double max_similarity =
                    max_similarity_to_selected[static_cast<size_t>(qi)];
                if (max_similarity < best_max_similarity) {
                    best_max_similarity = max_similarity;
                    best = static_cast<int32_t>(qi);
                }
            }
            onebit_query_order[static_cast<size_t>(pos)] = best;
            selected[static_cast<size_t>(best)] = 1;
            for (npy_intp qi = 0; qi < q_len; ++qi) {
                if (selected[static_cast<size_t>(qi)] != 0) {
                    continue;
                }
                double& max_similarity =
                    max_similarity_to_selected[static_cast<size_t>(qi)];
                max_similarity = std::max(
                    max_similarity,
                    query_cosines[
                        static_cast<size_t>(qi) * static_cast<size_t>(q_len) +
                        static_cast<size_t>(best)]);
            }
        }
    }
    const auto query_sim_precompute_end = Clock::now();

    auto collect_centroids_start = Clock::now();
    thread_local std::vector<uint16_t> used_centroids;
    used_centroids.clear();
    thread_local std::vector<uint16_t> center_lut;
    const size_t center_lut_size = static_cast<size_t>(n_centroids);
    if (center_lut.size() < center_lut_size) {
        center_lut.resize(center_lut_size);
    }
    uint16_t* center_lut_data = center_lut.data();
    thread_local std::vector<uint64_t> center_seen_words;
    const size_t seen_word_count = (center_lut_size + 63) / 64;
    if (center_seen_words.size() < seen_word_count) {
        center_seen_words.assign(seen_word_count, 0);
    }
    used_centroids.reserve(std::min(static_cast<size_t>(n_centroids), static_cast<size_t>(n_cands) * 64));
    const int union_thread_count = n_cands >= 500
        ? std::min(omp_get_max_threads(), static_cast<int>(n_cands))
        : 1;
    if (union_thread_count > 1) {
        thread_local std::vector<uint64_t> thread_seen_words;
        const size_t thread_words_size =
            static_cast<size_t>(union_thread_count) * seen_word_count;
        thread_seen_words.resize(thread_words_size);
        std::fill(thread_seen_words.begin(), thread_seen_words.end(), uint64_t{0});
        uint64_t* thread_seen_data = thread_seen_words.data();
#pragma omp parallel num_threads(union_thread_count)
        {
            uint64_t* local_seen = thread_seen_data +
                static_cast<size_t>(omp_get_thread_num()) * seen_word_count;
#pragma omp for schedule(static)
            for (npy_intp c = 0; c < n_cands; ++c) {
                const int64_t doc_id = candidates[c];
                const size_t centroid_begin = static_cast<size_t>(doc_centroid_offsets[doc_id]);
                const size_t centroid_end = static_cast<size_t>(doc_centroid_offsets[doc_id + 1]);
                for (size_t pos = centroid_begin; pos < centroid_end; ++pos) {
                    const uint16_t centroid_id = doc_hit_centroids[pos];
                    local_seen[static_cast<size_t>(centroid_id) >> 6] |=
                        uint64_t{1} << (static_cast<size_t>(centroid_id) & 63);
                }
            }
        }
        for (size_t word_i = 0; word_i < seen_word_count; ++word_i) {
            uint64_t bits = 0;
            for (int thread_i = 0; thread_i < union_thread_count; ++thread_i) {
                bits |= thread_seen_data[
                    static_cast<size_t>(thread_i) * seen_word_count + word_i];
            }
            center_seen_words[word_i] = bits;
        }
    } else {
        for (npy_intp c = 0; c < n_cands; ++c) {
            const int64_t doc_id = candidates[c];
            const size_t centroid_begin = static_cast<size_t>(doc_centroid_offsets[doc_id]);
            const size_t centroid_end = static_cast<size_t>(doc_centroid_offsets[doc_id + 1]);
            for (size_t pos = centroid_begin; pos < centroid_end; ++pos) {
                const uint16_t centroid_id = doc_hit_centroids[pos];
                center_seen_words[static_cast<size_t>(centroid_id) >> 6] |=
                    uint64_t{1} << (static_cast<size_t>(centroid_id) & 63);
            }
        }
    }
    for (size_t word_i = 0; word_i < seen_word_count; ++word_i) {
        uint64_t bits = center_seen_words[word_i];
        center_seen_words[word_i] = 0;
        while (bits != 0) {
            const unsigned bit_i = static_cast<unsigned>(__builtin_ctzll(bits));
            const size_t center_i = (word_i << 6) + bit_i;
            if (center_i < center_lut_size) {
                if (used_centroids.size() >= static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
                    PyErr_SetString(PyExc_ValueError, "a query uses too many IVF centers for the uint16 local-center LUT");
                    return nullptr;
                }
                center_lut_data[center_i] = static_cast<uint16_t>(used_centroids.size());
                used_centroids.push_back(static_cast<uint16_t>(center_i));
            }
            bits &= bits - 1;
        }
    }

    thread_local std::vector<size_t> candidate_token_offsets_storage;
    thread_local std::vector<uint32_t> candidate_center_row_offsets_storage;
    const size_t candidate_center_table_size =
        static_cast<size_t>(q_len) * used_centroids.size();
    if (candidate_center_table_size != 0 &&
        candidate_center_table_size - 1 >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        PyErr_SetString(PyExc_ValueError, "query/local-center table is too large for uint32 row offsets");
        return nullptr;
    }

    candidate_token_offsets_storage.resize(static_cast<size_t>(n_cands) + 1);
    candidate_token_offsets_storage[0] = 0;
    for (npy_intp c = 0; c < n_cands; ++c) {
        const int64_t doc_id = candidates[c];
        const size_t token_begin = static_cast<size_t>(doc_offsets[doc_id]);
        const size_t token_end = static_cast<size_t>(doc_offsets[doc_id + 1]);
        candidate_token_offsets_storage[static_cast<size_t>(c) + 1] =
            candidate_token_offsets_storage[static_cast<size_t>(c)] +
            (token_end - token_begin);
    }
    candidate_center_row_offsets_storage.resize(candidate_token_offsets_storage.back());
    const size_t* candidate_offsets_build_data = candidate_token_offsets_storage.data();
    uint32_t* candidate_rows_build_data = candidate_center_row_offsets_storage.data();
#pragma omp parallel for schedule(static) if(candidate_token_offsets_storage.back() >= 32768)
    for (npy_intp c = 0; c < n_cands; ++c) {
        const int64_t doc_id = candidates[c];
        const size_t token_begin = static_cast<size_t>(doc_offsets[doc_id]);
        const size_t token_end = static_cast<size_t>(doc_offsets[doc_id + 1]);
        const size_t output_begin =
            candidate_offsets_build_data[static_cast<size_t>(c)];
        for (size_t token_global = token_begin; token_global < token_end; ++token_global) {
            const uint16_t local_center =
                center_lut_data[static_cast<size_t>(centroid_ids[token_global])];
            candidate_rows_build_data[output_begin + (token_global - token_begin)] =
                static_cast<uint32_t>(
                    static_cast<size_t>(local_center) * static_cast<size_t>(q_len));
        }
    }
    const size_t* candidate_token_offsets_data = candidate_token_offsets_storage.data();
    const uint32_t* candidate_center_row_offsets_data = candidate_center_row_offsets_storage.data();

    auto collect_centroids_end = Clock::now();

    auto precompute_q_center_start = Clock::now();
    const size_t used_center_count = used_centroids.size();
    const uint16_t* used_centroids_data = used_centroids.data();
    const size_t center_table_size = static_cast<size_t>(q_len) * used_center_count;
    if (center_table_size != 0 &&
        center_table_size - 1 > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        PyErr_SetString(PyExc_ValueError, "query/local-center table is too large for uint32 row offsets");
        return nullptr;
    }
    std::unique_ptr<float[]> q_dot_centers(new float[center_table_size]);
    std::unique_ptr<float[]> g_errors(new float[center_table_size]);
    if (q_len > std::numeric_limits<int>::max() ||
        used_center_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        dim > std::numeric_limits<int>::max()) {
        PyErr_SetString(PyExc_ValueError, "rerank_topk_rabitq dimensions exceed BLAS int limits");
        return nullptr;
    }

    std::unique_ptr<float[]> q_norm_sq(new float[static_cast<size_t>(q_len)]);
#pragma omp parallel for schedule(static)
    for (npy_intp qi = 0; qi < q_len; ++qi) {
        const float* q_ptr = q + (static_cast<size_t>(qi) * static_cast<size_t>(dim));
        double norm_sq = 0.0;
        for (int d = 0; d < dim; ++d) {
            norm_sq += static_cast<double>(q_ptr[d]) * static_cast<double>(q_ptr[d]);
        }
        q_norm_sq[static_cast<size_t>(qi)] = std::nextafter(
            static_cast<float>(norm_sq), std::numeric_limits<float>::infinity());
    }
#pragma omp parallel for schedule(static)
    for (npy_intp local_center_i = 0; local_center_i < static_cast<npy_intp>(used_center_count); ++local_center_i) {
        const size_t local_center = static_cast<size_t>(local_center_i);
        const uint16_t center_id = used_centroids_data[local_center];
        const float* center_ptr = centroids + (static_cast<size_t>(center_id) * static_cast<size_t>(dim));
        const float center_norm_sq = centroid_norm_sq[static_cast<size_t>(center_id)];
        for (npy_intp qi_i = 0; qi_i < q_len; ++qi_i) {
            const size_t qi = static_cast<size_t>(qi_i);
            const float* q_ptr = q + (qi * static_cast<size_t>(dim));
            float q_dot_center = 0.0F;
#pragma omp simd reduction(+:q_dot_center)
            for (int d = 0; d < dim; ++d) {
                q_dot_center += q_ptr[d] * center_ptr[d];
            }
            const size_t pair_idx = (local_center * static_cast<size_t>(q_len)) + qi;
            q_dot_centers[pair_idx] = q_dot_center;
            float dist_sq = q_norm_sq[qi] + center_norm_sq - (2.0F * q_dot_center);
            if (dist_sq < 0.0F) {
                dist_sq = 0.0F;
            }
            g_errors[pair_idx] = std::sqrt(dist_sq);
        }
    }
    auto precompute_q_center_end = Clock::now();

    std::unique_ptr<LockFreeTopKScores> onebit_topk_lower_bounds;
    onebit_topk_lower_bounds = std::make_unique<LockFreeTopKScores>(static_cast<size_t>(top_k));
    int64_t onebit_total_blocks = 0;
    int64_t onebit_scanned_blocks = 0;
    int64_t onebit_query_skip_doc_count = 0;
    int64_t nbit_pair_eval_count = 0;
    int64_t nbit_pair_total_count = 0;
    int64_t nbit_query_skip_doc_count = 0;
    const bool NBIT_QUERY_BOUNDS = total_bits > 1;
    std::vector<float> onebit_query_uppers(
        NBIT_QUERY_BOUNDS
            ? static_cast<size_t>(n_cands) * static_cast<size_t>(q_len)
            : 0, 1.0F);
    std::vector<float> maxsim_scores(static_cast<size_t>(n_cands));

    // Scores candidates with one-bit bounds and online document pruning.
    auto score_docs_1bit = [&](const int32_t* positions,
                               size_t position_count,
                               std::vector<float>* maxsim_lowers,
                               std::vector<float>* maxsim_uppers) {
        const bool NEED_SCORE = total_bits == 1;
        Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic) reduction(+:onebit_total_blocks, onebit_scanned_blocks, onebit_query_skip_doc_count)
        for (npy_intp pi = 0; pi < static_cast<npy_intp>(position_count); ++pi) {
            const int32_t c = positions[static_cast<size_t>(pi)];
            const int64_t doc_id = candidates[c];
            float maxsim_est = 0.0F;
            float maxsim_lower = 0.0F;
            float maxsim_upper = 0.0F;
            const size_t token_begin = static_cast<size_t>(doc_offsets[doc_id]);
            const size_t token_end = static_cast<size_t>(doc_offsets[doc_id + 1]);
            const size_t token_count = token_end - token_begin;
            const size_t batch_begin = static_cast<size_t>(doc_batch_offsets[doc_id]);
            const size_t batch_count = static_cast<size_t>(doc_batch_offsets[doc_id + 1] - doc_batch_offsets[doc_id]);
            const uint32_t* doc_center_row_offsets = candidate_center_row_offsets_data +
                candidate_token_offsets_data[static_cast<size_t>(c)];
            const float doc_prune_threshold = onebit_topk_lower_bounds->tau();
            int64_t local_onebit_scanned_blocks = 0;
            bool DOC_PRUNED = false;
            float doc_partial_maxsim_upper = std::numeric_limits<float>::infinity();
            double remaining_query_upper_sum = static_cast<double>(q_len);
            thread_local std::vector<float> remaining_query_uppers;
            remaining_query_uppers.assign(static_cast<size_t>(q_len), 1.0F);
            for (npy_intp query_pos = 0; query_pos < q_len; ++query_pos) {
                const npy_intp qi = static_cast<npy_intp>(
                    onebit_query_order[static_cast<size_t>(query_pos)]);
                remaining_query_upper_sum -= remaining_query_uppers[static_cast<size_t>(qi)];
                const auto& query = split_queries[static_cast<size_t>(qi)];
                float max_ip_est = -std::numeric_limits<float>::infinity();
                float query_mip_max = -std::numeric_limits<float>::infinity();
                float query_mip_min = -std::numeric_limits<float>::infinity();
                for (size_t block_pos = 0; block_pos < batch_count; ++block_pos) {
                    const size_t b = block_pos;
                    const char* batch_ptr = onebit_data + ((batch_begin + b) * batch_bytes);
                    ++local_onebit_scanned_blocks;
                    std::array<float, rabitqlib::fastscan::kBatchSize> est_distance;
                    split_batch_estdist_no_low(
                        batch_ptr,
                        query,
                        static_cast<size_t>(dim),
                        est_distance.data(),
                        use_hacc);

                    rabitqlib::ConstBatchDataMap<float> cur_batch(batch_ptr, static_cast<size_t>(dim));
                    const size_t local_token_base = b * rabitqlib::fastscan::kBatchSize;
                    const size_t remaining = token_count - local_token_base;
                    const size_t valid_in_batch = std::min(rabitqlib::fastscan::kBatchSize, remaining);
                    for (size_t i = 0; i < valid_in_batch; ++i) {
                        const size_t center_offset =
                            static_cast<size_t>(doc_center_row_offsets[local_token_base + i]) +
                            static_cast<size_t>(qi);
                        const float q_dot_center = q_dot_centers[center_offset];

                        const float ip_est = 1.0F - est_distance[i] + q_dot_center;
                        if (NEED_SCORE && ip_est > max_ip_est) {
                            max_ip_est = ip_est;
                        }
                        const float ip_bound = cur_batch.f_error()[i] * g_errors[center_offset];
                        const float mip_max = ip_est + ip_bound;
                        const float mip_min = ip_est - ip_bound;
                        if (mip_max > query_mip_max) {
                            query_mip_max = mip_max;
                        }
                        if (mip_min > query_mip_min) {
                            query_mip_min = mip_min;
                        }
                    }
                }
                if (NEED_SCORE) {
                    maxsim_est += max_ip_est;
                }
                maxsim_upper += query_mip_max;
                maxsim_lower += query_mip_min;
                if (NBIT_QUERY_BOUNDS) {
                    const size_t query_idx = static_cast<size_t>(c) * static_cast<size_t>(q_len) + static_cast<size_t>(qi);
                    onebit_query_uppers[query_idx] = std::min(1.0F, query_mip_max);
                }
                if (std::isfinite(doc_prune_threshold)) {
                    const double processed_upper = static_cast<double>(maxsim_upper);
                    const double ip_upper = std::max(-1.0, std::min(1.0, static_cast<double>(query_mip_max)));
                    const double ip_orthogonal_sine = std::sqrt(std::max(0.0, 1.0 - ip_upper * ip_upper));
                    for (npy_intp future_pos = query_pos + 1; future_pos < q_len; ++future_pos) {
                        const int32_t qj = onebit_query_order[static_cast<size_t>(future_pos)];
                        const double cosine = query_cosines[
                            static_cast<size_t>(qi) * static_cast<size_t>(q_len) + static_cast<size_t>(qj)];
                        if (cosine <= ip_upper) {
                            continue;
                        }
                        const double safe_geometric_upper =
                            cosine * ip_upper +
                            query_orthogonal_sines[
                                static_cast<size_t>(qi) * static_cast<size_t>(q_len) + static_cast<size_t>(qj)] *
                            ip_orthogonal_sine;
                        const double relaxed_geometric_upper =
                            ip_upper +
                            alpha_geometry * (safe_geometric_upper - ip_upper);
                        const float propagated_upper = std::nextafter(
                            static_cast<float>(std::min(1.0, relaxed_geometric_upper)),
                            std::numeric_limits<float>::infinity());
                        float& current_upper = remaining_query_uppers[static_cast<size_t>(qj)];
                        if (propagated_upper < current_upper) {
                            remaining_query_upper_sum -= static_cast<double>(current_upper - propagated_upper);
                            current_upper = propagated_upper;
                        }
                    }
                    doc_partial_maxsim_upper = std::nextafter(
                        static_cast<float>(processed_upper + remaining_query_upper_sum),
                        std::numeric_limits<float>::infinity());
                    if (doc_partial_maxsim_upper < doc_prune_threshold) {
                        DOC_PRUNED = true;
                        break;
                    }
                }
            }

            if (NEED_SCORE) {
                maxsim_scores[static_cast<size_t>(c)] = maxsim_est;
            }
            (*maxsim_lowers)[static_cast<size_t>(c)] = DOC_PRUNED
                ? -std::numeric_limits<float>::infinity()
                : maxsim_lower;
            (*maxsim_uppers)[static_cast<size_t>(c)] = DOC_PRUNED
                ? doc_partial_maxsim_upper
                : maxsim_upper;
            if (!DOC_PRUNED) {
                const float document_lower = (*maxsim_lowers)[static_cast<size_t>(c)];
                onebit_topk_lower_bounds->insert(document_lower);
            }
            onebit_total_blocks += static_cast<int64_t>(q_len) * static_cast<int64_t>(batch_count);
            onebit_scanned_blocks += local_onebit_scanned_blocks;
            if (DOC_PRUNED) {
                ++onebit_query_skip_doc_count;
            }
        }
        Py_END_ALLOW_THREADS
    };

    // Refines surviving candidates with extra bits and query-level pruning.
    auto score_docs_nbit = [&](const int32_t* positions, size_t position_count, float maxsim_prune_threshold) {
        Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(dynamic) reduction(+:nbit_query_skip_doc_count)
        for (npy_intp pi = 0; pi < static_cast<npy_intp>(position_count); ++pi) {
            const int32_t c = positions[static_cast<size_t>(pi)];
            const int64_t doc_id = candidates[c];
            const size_t token_begin = static_cast<size_t>(doc_offsets[doc_id]);
            const size_t token_end = static_cast<size_t>(doc_offsets[doc_id + 1]);
            const size_t token_count = token_end - token_begin;
            const size_t batch_begin = static_cast<size_t>(doc_batch_offsets[doc_id]);
            const size_t batch_count = static_cast<size_t>(doc_batch_offsets[doc_id + 1] - doc_batch_offsets[doc_id]);
            const uint32_t* doc_center_row_offsets = candidate_center_row_offsets_data +
                candidate_token_offsets_data[static_cast<size_t>(c)];
            int64_t local_nbit_pair_eval = 0;
            float maxsim_est = 0.0F;
            bool DOC_PRUNED = false;

            thread_local std::vector<float> remaining_query_mip_uppers;
            const bool USE_QUERY_DOCUMENT_EARLY_TERMINATION =
                NBIT_QUERY_BOUNDS && std::isfinite(maxsim_prune_threshold);
            double remaining_maxsim_upper = 0.0;
            double processed_maxsim = 0.0;

            if (USE_QUERY_DOCUMENT_EARLY_TERMINATION) {
                const size_t query_row = static_cast<size_t>(c) * static_cast<size_t>(q_len);
                remaining_query_mip_uppers.assign(onebit_query_uppers.begin() + static_cast<std::ptrdiff_t>(query_row), onebit_query_uppers.begin() + static_cast<std::ptrdiff_t>(query_row + static_cast<size_t>(q_len)));
                for (npy_intp qi = 0; qi < q_len; ++qi) {
                    remaining_maxsim_upper += remaining_query_mip_uppers[static_cast<size_t>(qi)];
                }
            }

            for (npy_intp query_pos = 0; query_pos < q_len; ++query_pos) {
                const npy_intp qi = static_cast<npy_intp>(
                    onebit_query_order[static_cast<size_t>(query_pos)]);
                const auto& query = split_queries[static_cast<size_t>(qi)];
                float max_ip_est = -std::numeric_limits<float>::infinity();

                for (size_t block_pos = 0; block_pos < batch_count; ++block_pos) {
                    const char* batch_ptr = onebit_data + ((batch_begin + block_pos) * batch_bytes);
                    std::array<float, rabitqlib::fastscan::kBatchSize> est_distance;
                    std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr;
                    split_batch_estdist_no_low(
                        batch_ptr,
                        query,
                        static_cast<size_t>(dim),
                        est_distance.data(),
                        ip_x0_qr.data(),
                        use_hacc);

                    rabitqlib::ConstBatchDataMap<float> cur_batch(batch_ptr, static_cast<size_t>(dim));
                    const size_t local_token_base = block_pos * rabitqlib::fastscan::kBatchSize;
                    const size_t valid_in_batch = std::min(rabitqlib::fastscan::kBatchSize, token_count - local_token_base);
                    for (size_t i = 0; i < valid_in_batch; ++i) {
                        const size_t token_global = token_begin + local_token_base + i;
                        const size_t center_offset =
                            static_cast<size_t>(doc_center_row_offsets[local_token_base + i]) +
                            static_cast<size_t>(qi);
                        const float q_dot_center = q_dot_centers[center_offset];
                        const float onebit_ip_est = 1.0F - est_distance[i] + q_dot_center;
                        const float onebit_ip_bound = cur_batch.f_error()[i] * g_errors[center_offset];
                        const float nbit_ip_bound = onebit_ip_bound / static_cast<float>(1u << ex_bits);
                        const float gate_ip_upper = onebit_ip_est + nbit_ip_bound;
                        if (gate_ip_upper <= max_ip_est) {
                            continue;
                        }
                        const char* ex_ptr = ex_data + (token_global * ex_stride);
                        const float ip_est = 1.0F - rabitqlib::split_distance_boosting(
                                            ex_ptr,
                                            ip_func,
                                            query,
                                            static_cast<size_t>(dim),
                                            ex_bits,
                                            ip_x0_qr[i]) + q_dot_center;
                        ++local_nbit_pair_eval;
                        if (ip_est > max_ip_est) {
                            max_ip_est = ip_est;
                        }
                    }
                }
                maxsim_est += max_ip_est;

                if (USE_QUERY_DOCUMENT_EARLY_TERMINATION) {
                    processed_maxsim += static_cast<double>(max_ip_est);
                    remaining_maxsim_upper -= remaining_query_mip_uppers[static_cast<size_t>(qi)];
                    if (processed_maxsim + remaining_maxsim_upper < static_cast<double>(maxsim_prune_threshold)) {
                        DOC_PRUNED = true;
                        ++nbit_query_skip_doc_count;
                        break;
                    }
                }
            }

            maxsim_scores[static_cast<size_t>(c)] = DOC_PRUNED ? -std::numeric_limits<float>::infinity() : maxsim_est;
            if (local_nbit_pair_eval > 0) {
#pragma omp atomic
                nbit_pair_eval_count += local_nbit_pair_eval;
            }
        }
        Py_END_ALLOW_THREADS
    };


    std::vector<int32_t> candidate_doc_idx(static_cast<size_t>(n_cands));
    for (npy_intp i = 0; i < n_cands; ++i) {
        candidate_doc_idx[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    std::vector<float> maxsim_lowers(
        static_cast<size_t>(n_cands), -std::numeric_limits<float>::infinity());
    std::vector<float> maxsim_uppers(
        static_cast<size_t>(n_cands), -std::numeric_limits<float>::infinity());
    std::vector<int32_t> remained_doc_idx;
    std::vector<int32_t> refined_positions;
    constexpr size_t kRefineBatchSize = 64;

    const auto score_start = Clock::now();
    score_docs_1bit(
        candidate_doc_idx.data(), static_cast<size_t>(top_k),
        &maxsim_lowers, &maxsim_uppers);
    score_docs_1bit(
        candidate_doc_idx.data() + top_k,
        candidate_doc_idx.size() - static_cast<size_t>(top_k),
        &maxsim_lowers, &maxsim_uppers);
    const auto score_end = Clock::now();

    const auto select_remained_doc_start = Clock::now();
    std::vector<float> lower_copy = maxsim_lowers;
    std::nth_element(
        lower_copy.begin(), lower_copy.begin() + (top_k - 1),
        lower_copy.end(), std::greater<float>());
    const float tau_k = lower_copy[static_cast<size_t>(top_k - 1)];
    remained_doc_idx.reserve(static_cast<size_t>(n_cands));
    for (npy_intp i = 0; i < n_cands; ++i) {
        if (maxsim_uppers[static_cast<size_t>(i)] >= tau_k) {
            remained_doc_idx.push_back(static_cast<int32_t>(i));
        }
    }
    std::sort(
        remained_doc_idx.begin(), remained_doc_idx.end(),
        [&](int32_t lhs, int32_t rhs) {
            const float lhs_upper = maxsim_uppers[static_cast<size_t>(lhs)];
            const float rhs_upper = maxsim_uppers[static_cast<size_t>(rhs)];
            return lhs_upper > rhs_upper ||
                (lhs_upper == rhs_upper && lhs < rhs);
        });
    std::vector<float> remaining_upper_suffix(remained_doc_idx.size() + 1,
        -std::numeric_limits<float>::infinity());
    for (size_t i = remained_doc_idx.size(); i > 0; --i) {
        remaining_upper_suffix[i - 1] = std::max(
            remaining_upper_suffix[i],
            maxsim_uppers[static_cast<size_t>(remained_doc_idx[i - 1])]);
    }
    const auto select_remained_doc_end = Clock::now();

    for (const int32_t c : remained_doc_idx) {
        const int64_t doc_id = candidates[c];
        nbit_pair_total_count += static_cast<int64_t>(q_len) *
            static_cast<int64_t>(doc_offsets[doc_id + 1] - doc_offsets[doc_id]);
    }
    refined_positions.reserve(remained_doc_idx.size());
    std::priority_queue<float, std::vector<float>, std::greater<float>> refined_topk_scores;
    const bool MAINTAIN_REFINED_TOPK =
        remained_doc_idx.size() > static_cast<size_t>(top_k);
    const auto refine_start = Clock::now();
    if (total_bits > 1 && !remained_doc_idx.empty()) {
        for (size_t start = 0; start < remained_doc_idx.size(); start += kRefineBatchSize) {
            const size_t end = std::min(start + kRefineBatchSize, remained_doc_idx.size());
            const float maxsim_prune_threshold =
                refined_topk_scores.size() == static_cast<size_t>(top_k)
                ? refined_topk_scores.top()
                : -std::numeric_limits<float>::infinity();
            score_docs_nbit(
                remained_doc_idx.data() + start, end - start,
                maxsim_prune_threshold);
            for (size_t pos_idx = start; pos_idx < end; ++pos_idx) {
                const int32_t pos = remained_doc_idx[pos_idx];
                if (std::isfinite(maxsim_scores[static_cast<size_t>(pos)])) {
                    refined_positions.push_back(pos);
                }
            }

            if (MAINTAIN_REFINED_TOPK) {
                for (size_t pos_idx = start; pos_idx < end; ++pos_idx) {
                    const int32_t pos = remained_doc_idx[pos_idx];
                    const float score = maxsim_scores[static_cast<size_t>(pos)];
                    if (!std::isfinite(score)) {
                        continue;
                    }
                    if (refined_topk_scores.size() < static_cast<size_t>(top_k)) {
                        refined_topk_scores.push(score);
                    } else if (score > refined_topk_scores.top()) {
                        refined_topk_scores.pop();
                        refined_topk_scores.push(score);
                    }
                }
            }

            if (refined_positions.size() >= static_cast<size_t>(top_k) &&
                end < remained_doc_idx.size()) {
                const float kth_refined_score = refined_topk_scores.top();
                const float next_unrefined_upper = remaining_upper_suffix[end];
                if (next_unrefined_upper < kth_refined_score) {
                    break;
                }
            }
        }
    } else {
        refined_positions = remained_doc_idx;
    }
    const auto refine_end = Clock::now();

    const size_t result_count = std::min(static_cast<size_t>(top_k), refined_positions.size());
    if (refined_positions.size() > result_count) {
        std::nth_element(
            refined_positions.begin(),
            refined_positions.begin() + static_cast<std::ptrdiff_t>(result_count),
            refined_positions.end(),
                [&](int32_t lhs, int32_t rhs) {
                    const float lhs_score = maxsim_scores[static_cast<size_t>(lhs)];
                    const float rhs_score = maxsim_scores[static_cast<size_t>(rhs)];
                return lhs_score > rhs_score || (lhs_score == rhs_score && lhs < rhs);
            });
    }
    std::sort(
        refined_positions.begin(),
        refined_positions.begin() + static_cast<std::ptrdiff_t>(result_count),
        [&](int32_t lhs, int32_t rhs) {
            const float lhs_score = maxsim_scores[static_cast<size_t>(lhs)];
            const float rhs_score = maxsim_scores[static_cast<size_t>(rhs)];
            return lhs_score > rhs_score || (lhs_score == rhs_score && lhs < rhs);
        });

    npy_intp out_dims[1] = {static_cast<npy_intp>(result_count)};
    ArrayRef sorted_arr(reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, out_dims, NPY_INT64)));
    if (!sorted_arr) {
        return nullptr;
    }
    auto* sorted_positions = static_cast<int64_t*>(PyArray_DATA(sorted_arr.get()));
    for (size_t i = 0; i < result_count; ++i) {
        sorted_positions[i] = refined_positions[i];
    }

    const double onebit_score_time_s =
        std::chrono::duration<double>(score_end - score_start).count();
    const double query_lut_build_time_s =
        std::chrono::duration<double>(
            query_lut_build_end - query_lut_build_start).count();
    const double query_sim_precompute_time_s =
        std::chrono::duration<double>(
            query_sim_precompute_end - query_sim_precompute_start).count();
    const double collect_used_centroids_time_s =
        std::chrono::duration<double>(collect_centroids_end - collect_centroids_start).count();
    const double precompute_q_center_time_s =
        std::chrono::duration<double>(precompute_q_center_end - precompute_q_center_start).count();
    const double select_survivors_time_s =
        std::chrono::duration<double>(select_remained_doc_end - select_remained_doc_start).count();
    const double nbit_refine_time_s =
        std::chrono::duration<double>(refine_end - refine_start).count();
    const double nbit_pair_skip_ratio = nbit_pair_total_count > 0
        ? 1.0 - (static_cast<double>(nbit_pair_eval_count) /
                 static_cast<double>(nbit_pair_total_count))
        : 0.0;
    const double onebit_query_skip_doc_ratio =
        static_cast<double>(onebit_query_skip_doc_count) /
        static_cast<double>(n_cands);
    const double onebit_fastscan_reduction_ratio = onebit_total_blocks > 0
        ? 1.0 - (
              static_cast<double>(onebit_scanned_blocks) /
              static_cast<double>(onebit_total_blocks))
        : 0.0;
    PyObject* stats = Py_BuildValue(
        "{s:d,s:d,s:d,s:d,s:d,s:d,s:L,s:L,s:L,s:d,s:d,s:L,s:d,s:d}",
        "1bit_score_time_s",
        onebit_score_time_s,
        "query_lut_build_time_s",
        query_lut_build_time_s,
        "query_sim_precompute_time_s",
        query_sim_precompute_time_s,
        "collect_used_centroids_time_s",
        collect_used_centroids_time_s,
        "precompute_q_center_time_s",
        precompute_q_center_time_s,
        "select_survivors_time_s",
        select_survivors_time_s,
        "1bit_pruned_doc_count",
        static_cast<long long>(n_cands - static_cast<npy_intp>(remained_doc_idx.size())),
        "nbit_doc_count",
        static_cast<long long>(remained_doc_idx.size()),
        "nbit_refined_doc_count",
        static_cast<long long>(refined_positions.size()),
        "nbit_refine_time_s",
        nbit_refine_time_s,
        "nbit_pair_skip_ratio",
        nbit_pair_skip_ratio,
        "nbit_query_early_pruned_doc_count",
        static_cast<long long>(nbit_query_skip_doc_count),
        "onebit_fastscan_reduction_ratio",
        onebit_fastscan_reduction_ratio,
        "1bit_query_skip_doc_ratio",
        onebit_query_skip_doc_ratio);

    return Py_BuildValue("NN", sorted_arr.release_object(), stats);
}

PyMethodDef methods[] = {
    {"encode_pack_rabitq", reinterpret_cast<PyCFunction>(encode_pack_rabitq), METH_VARARGS | METH_KEYWORDS,
     "Encode rotated vectors and pack them directly into FastScan and ExDataMap layouts."},
    {"rerank_topk_rabitq", reinterpret_cast<PyCFunction>(rerank_topk_rabitq), METH_VARARGS | METH_KEYWORDS,
     "Run 1-bit bound pruning, optional n-bit refine, and top-k selection in C++."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "rabitq_official_ext",
    "Official RaBitQ-Library Python wrapper.",
    -1,
    methods,
};

}

PyMODINIT_FUNC PyInit_rabitq_official_ext() {
    import_array();
    PyObject* module_obj = PyModule_Create(&module);
    if (module_obj == nullptr) {
        return nullptr;
    }
    if (PyModule_AddIntConstant(module_obj, "RERANK_ABLATION_API_VERSION", 3) < 0) {
        Py_DECREF(module_obj);
        return nullptr;
    }
    return module_obj;
}
