
#include "graph.hpp"

#include <algorithm>
#include <unordered_map>

#include "event_tracer.hpp"
#include "ggml-impl.h"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
#    define GRAPH_PROFILE_HANDLE (_event_tracer ? _event_tracer->get_handle() : nullptr)
#    define GRAPH_PROFILE_PRINT()                  \
        if (_event_tracer) {                       \
            _event_tracer->print_profile_events(); \
        }                                          \
        (void) 0
#else
#    define GRAPH_PROFILE_HANDLE  (nullptr)
#    define GRAPH_PROFILE_PRINT() (void) 0
#endif

namespace {
using qnn_tensor_cache_t = std::unordered_map<ggml_tensor *, qnn::qnn_tensor_ptr_t>;

int get_op_max_rank(const ggml_tensor * op) {
    int max_rank = ggml_n_dims(op);
    for (int i = 0; i < GGML_MAX_DIMS && op->src[i]; ++i) {
        max_rank = std::max(max_rank, ggml_n_dims(op->src[i]));
    }

    return max_rank;
}

qnn::qnn_tensor_ptr_t create_tensor_with_cache(ggml_tensor * tensor, qnn::ggml_qnn_tensor::tensor_type_t type, int rank,
                                               ggml_type override_data_type, backend_index_type device,
                                               Qnn_GraphHandle_t                  graph_handle,
                                               std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                               qnn_tensor_cache_t &               tensor_cache) {
    GGML_ASSERT(tensor);
    if (tensor_cache.count(tensor)) {
        return tensor_cache[tensor];
    }

    QNN_LOG_DEBUG("[%s]create_tensor_with_cache, data_type: %s, override_data_type: %s\n",
                  qnn::get_backend_name(device), ggml_type_name(tensor->type), ggml_type_name(override_data_type));
    auto data_type = override_data_type != GGML_TYPE_COUNT ? override_data_type : tensor->type;

    // We've observed that some tensors have the same name with different op types will be added to the same graph
    // which will cause the graph build failed. To avoid this, we append the op type to the tensor name.
    char tensor_name[256];
    snprintf(tensor_name, sizeof(tensor_name), "%s_%s", ggml_get_name(tensor), ggml_op_desc(tensor));
    auto qnn_tensor      = std::make_shared<qnn::ggml_qnn_tensor>(type, std::string(tensor_name), tensor->ne, data_type,
                                                                  rank, device, graph_handle, qnn_instance);
    tensor_cache[tensor] = qnn_tensor;
    return qnn_tensor;
}

qnn::qnn_tensor_array_t create_tensors_with_cache(const qnn::ggml_tensor_array_t &    ggml_tensors,
                                                  qnn::ggml_qnn_tensor::tensor_type_t type, int rank,
                                                  ggml_type override_data_type, backend_index_type device,
                                                  Qnn_GraphHandle_t                  graph_handle,
                                                  std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                  qnn_tensor_cache_t &               tensor_cache) {
    qnn::qnn_tensor_array_t tensors;
    for (auto * tensor : ggml_tensors) {
        tensors.push_back(create_tensor_with_cache(tensor, type, rank, override_data_type, device, graph_handle,
                                                   qnn_instance, tensor_cache));
    }

    return tensors;
}

qnn::qnn_op_config_ptr_t create_operation_from_op_tensor(ggml_tensor * dst, const std::string & name, int rank,
                                                         backend_index_type device, Qnn_GraphHandle_t graph_handle,
                                                         std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                         qnn_tensor_cache_t &               tensor_cache) {
    auto operation = qnn::create_op(dst, name, qnn_instance);

    // input tensors
    qnn::qnn_tensor_array_t input_qnn_tensors;
    for (size_t i = 0; i < GGML_MAX_DIMS && dst->src[i]; ++i) {
        auto * src            = dst->src[i];
        auto input_qnn_tensor = create_tensor_with_cache(src, qnn::ggml_qnn_tensor::INTERMEDIATE, rank, GGML_TYPE_COUNT,
                                                         device, graph_handle, qnn_instance, tensor_cache);
        input_qnn_tensors.push_back(input_qnn_tensor);
    }
    operation->set_input_tensors(input_qnn_tensors);

    // output tensor
    qnn::qnn_tensor_array_t output_qnn_tensors =
        create_tensors_with_cache({ dst }, qnn::ggml_qnn_tensor::INTERMEDIATE, rank, GGML_TYPE_COUNT, device,
                                  graph_handle, qnn_instance, tensor_cache);
    operation->set_output_tensors(output_qnn_tensors);

    // initialize operation
    if (!operation->initialize_op_nodes(device, graph_handle)) {
        QNN_LOG_ERROR("[%s][%s]initialize_op_nodes failed\n", qnn::get_backend_name(device), name.c_str());
        return nullptr;
    }

    return operation;
}

/**
 * @brief Extracts input and output tensors from a computational graph.
 *
 * This function identifies the input and output tensors of a computational graph by analyzing the connectivity between
 * tensor nodes. It does this by iterating over each node in the graph, using a connectivity map that associates every
 * tensor with its number of incoming connections (in_degree), outgoing connections (out_degree), and an insertion index
 * that preserves order. The insertion index is used later to sort the tensors in their original discovery order.
 *
 * TODO: this algorithm is not perfect and may not work for all cases. It assumes that the tensors are
 *   connected in a way that allows for unambiguous categorization.
 */
int get_io_tensors_from_graph(const ggml_cgraph * cgraph, qnn::ggml_tensor_array_t & inputs,
                              qnn::ggml_tensor_array_t & outputs) {
    struct _tensor_connectivity_info {
        size_t in_degree    = 0;
        size_t out_degree   = 0;
        size_t insert_index = 0;
    };

    using ggml_tensor_connectivity_map_t = std::unordered_map<ggml_tensor *, _tensor_connectivity_info>;

    ggml_tensor_connectivity_map_t connectivity_map;
    int                            rank = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * dst = cgraph->nodes[i];
        if (ggml_is_empty(dst)) {
            continue;
        }

        if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW || dst->op == GGML_OP_PERMUTE) {
            // TODO: remove GGML_OP_VIEW after view op is supported
            QNN_LOG_DEBUG("node[%d]%s(%s), type: %s, skipped\n", i, ggml_get_name(dst), ggml_op_desc(dst),
                          ggml_type_name(dst->type));
            continue;
        }

        QNN_LOG_DEBUG("node[%d]%s(%s), type: %s\n", i, ggml_get_name(dst), ggml_op_desc(dst),
                      ggml_type_name(dst->type));
        rank = std::max(rank, ggml_n_dims(dst));
        if (connectivity_map.count(dst) == 0) {
            connectivity_map[dst] = {
                1,  // in-degree, at least 1
                0,
                connectivity_map.size(),
            };
        } else {
            ++(connectivity_map[dst].in_degree);
        }

        for (size_t j = 0; j < GGML_MAX_DIMS && dst->src[j]; ++j) {
            auto * src = dst->src[j];
            rank       = std::max(rank, ggml_n_dims(src));

            QNN_LOG_DEBUG("node[%d]: src[%d]: %s(%s), type: %s\n", i, (int) j, ggml_get_name(src), ggml_op_desc(src),
                          ggml_type_name(src->type));
            if (connectivity_map.count(src) == 0) {
                connectivity_map[src] = {
                    0,
                    1,  // out-degree, at least 1
                    connectivity_map.size(),
                };
            } else {
                ++(connectivity_map[src].out_degree);
            }
        }
    }

    for (const auto & kv : connectivity_map) {
        if (kv.second.in_degree == 0) {
            inputs.push_back(kv.first);
        }

        if (kv.second.out_degree == 0) {
            outputs.push_back(kv.first);
        }
    }

    std::sort(inputs.begin(), inputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    std::sort(outputs.begin(), outputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    return rank;
}

/*
 * for src0_F32, src1_F32, dst_F32 -> GGML_TYPE_COUNT
 * for src0_F16, src1_F16, dst_F16 -> GGML_TYPE_COUNT
 * for src0_F16, src1_F32, dst_F32 -> GGML_TYPE_F32
 * for src0_q4, src1_F32, dst_F32 -> GGML_TYPE_F32
 * for src0_q4, src1_F16, dst_F32 -> GGML_TYPE_F32
 */
ggml_type get_override_data_type(const qnn::ggml_tensor_array_t & inputs, const qnn::ggml_tensor_array_t & outputs,
                                 backend_index_type device) {
    GGML_ASSERT(!inputs.empty());
    ggml_type override_data_type = inputs.front()->type;
    bool      is_same_data_type  = true;
    for (auto * tensor : inputs) {
        QNN_LOG_DEBUG("input_tensor: %s(%s), override_data_type(%s)\n", ggml_get_name(tensor),
                      ggml_type_name(tensor->type), ggml_type_name(override_data_type));
        is_same_data_type  = is_same_data_type && tensor->type == override_data_type;
        override_data_type = std::min(override_data_type, tensor->type);
    }

    for (auto * tensor : outputs) {
        QNN_LOG_DEBUG("output_tensor: %s(%s), override_data_type(%s)\n", ggml_get_name(tensor),
                      ggml_type_name(tensor->type), ggml_type_name(override_data_type));
        is_same_data_type  = is_same_data_type && tensor->type == override_data_type;
        override_data_type = std::min(override_data_type, tensor->type);
    }

    if (is_same_data_type) {
        // [v68] an all-F32 NPU graph keeps F32 tensors while graph precision is
        // kHtpFp16, so libQnnHtpPrepare inserts Cast_fp32_to_fp16 (a v69+ only op)
        // and DSP-side graph activation fails. Override to F16 tensors instead so
        // the graph holds v68-native F16 matmuls with no Cast nodes.
        if (device == QNN_BACKEND_NPU && override_data_type == GGML_TYPE_F32) {
            bool all_float_tensors = true;
            for (auto * tensor : inputs) {
                all_float_tensors = all_float_tensors && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
            }
            for (auto * tensor : outputs) {
                all_float_tensors = all_float_tensors && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
            }
            if (all_float_tensors) {
                return GGML_TYPE_F16;
            }
        }
        return GGML_TYPE_COUNT;
    }

    // [v68] The HTP v68 skel has no Cast op implementations. A graph holding fp32
    // tensors forces libQnnHtpPrepare to insert Cast_fp32_to_fp16 (a v69+ only op)
    // which fails deserialization on the DSP. Keep float graphs in HTP-native F16
    // instead, converting fp32 activations on the host.
    if (device == QNN_BACKEND_NPU && override_data_type == GGML_TYPE_F32) {
        bool all_float_tensors = true;
        for (auto * tensor : inputs) {
            all_float_tensors = all_float_tensors && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
        }
        for (auto * tensor : outputs) {
            all_float_tensors = all_float_tensors && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
        }
        if (all_float_tensors) {
            return GGML_TYPE_F16;
        }
    }

    return override_data_type;
}

static const QnnHtpGraph_CustomConfig_t kDefaultHvxConfig = []() {
    QnnHtpGraph_CustomConfig_t hvx_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    hvx_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    hvx_config.numHvxThreads              = 8;
    return hvx_config;
}();

static const QnnHtpGraph_CustomConfig_t kDefaultDlbcConfig = []() {
    QnnHtpGraph_CustomConfig_t dlbc_config    = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    dlbc_config.option                        = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    dlbc_config.optimizationOption.type       = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
    // SA8295P v68: DLBC-flavored op variants (e.g. fp16_dual_input_1x1x1xd_add)
    // are not present in the HTP v68 skel op registry, graph deserialization on
    // the DSP fails with "op not found" -> disable DLBC.
    dlbc_config.optimizationOption.floatValue = 0.0;
    return dlbc_config;
}();

/*
 *  1 = Faster preparation time, less optimal graph
 *  2 = Longer preparation time, more optimal graph
 *  3 = Longest preparation time, most likely even more optimal graph:
 *   QNN_HTP_DEVICE_CONFIG_OPTION_SOC configuration will be taken into account when possible, details see HTP Backend Specific Page
 *
 * [v68] level 3 (SOC-optimized) pushes libQnnHtpPrepare into weight-packing /
 * graph-transformation paths (pack_fp16_pkweights_dynamic, ConvLayer.fp16) that
 * the v68 skel cannot deserialize. Level 1 keeps plain MatMul graphs, so it is
 * forced here regardless of build type. Override with QNN_HTP_OPT_LEVEL.
 */
static const QnnHtpGraph_CustomConfig_t kDefaultOptConfig = []() {
    QnnHtpGraph_CustomConfig_t opt_config    = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    opt_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    opt_config.optimizationOption.type    = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    {
        const char * opt_level_env = getenv("QNN_HTP_OPT_LEVEL");
        opt_config.optimizationOption.floatValue = opt_level_env ? atof(opt_level_env) : 1.0;
    }
    return opt_config;
}();

// [v68] libQnnHtpPrepare serializes a runtime pack_fp16_pkweights_dynamic node
// for dynamic F16 matmul weights when weight packing is enabled (a v69+ only
// op) — disable packing so dynamic weights stay a plain v68-native MatMul.
static const QnnHtpGraph_CustomConfig_t kDisableWeightsPackingConfig = []() {
    QnnHtpGraph_CustomConfig_t weights_packing = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    weights_packing.option          = QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING;
    weights_packing.weightsPacking  = false;
    return weights_packing;
}();

static const QnnHtpGraph_CustomConfig_t kHtpPrecisionConfigF16 = []() {
    QnnHtpGraph_CustomConfig_t precision_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    precision_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    precision_config.precision                  = QNN_PRECISION_FLOAT16;
    return precision_config;
}();

constexpr QnnHtpGraph_CustomConfig_t make_vtcm_config(size_t vtcm_size_in_mb) {
    QnnHtpGraph_CustomConfig_t vtcm_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    vtcm_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
    vtcm_config.vtcmSizeInMB               = (uint32_t) vtcm_size_in_mb;
    return vtcm_config;
}

constexpr QnnGraph_Config_t make_graph_config(const QnnHtpGraph_CustomConfig_t * custom_config) {
    QnnGraph_Config_t graph_config = QNN_GRAPH_CONFIG_INIT;
    graph_config.option            = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_config.customConfig      = const_cast<QnnHtpGraph_CustomConfig_t *>(custom_config);
    return graph_config;
}

}  // namespace

namespace qnn {

ggml_type qnn_graph::get_graph_key_from_cgraph(const ggml_cgraph * cgraph, std::string & output,
                                               backend_index_type device) {
    if (cgraph->n_nodes == 0) {
        QNN_LOG_DEBUG("empty cgraph\n");
        return GGML_TYPE_COUNT;
    }

    ggml_type override_type = GGML_TYPE_COUNT;
    {
        // TODO: can we have a better approach to get the override_type here?
        //   though it is O(n) + O(mlog(m)) complexity, our graph is small, so it is fine
        ggml_tensor_array_t inputs;
        ggml_tensor_array_t outputs;
        get_io_tensors_from_graph(cgraph, inputs, outputs);
        if (!inputs.empty() && !outputs.empty()) {
            override_type = get_override_data_type(inputs, outputs, device);
            QNN_LOG_DEBUG("get_graph_key, override_type: %s\n", ggml_type_name(override_type));
        } else {
            QNN_LOG_DEBUG("get_graph_key, no input or output tensors\n");
        }
    }

    ggml_type min_op_type = GGML_TYPE_COUNT;
    {
        bool is_start = true;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            auto * op = cgraph->nodes[i];
            if (ggml_is_empty(op)) {
                QNN_LOG_DEBUG("empty op in graph, skipping\n");
                continue;
            }

            if (op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE) {
                QNN_LOG_DEBUG("%s in graph, skipping\n", ggml_op_desc(op));
                continue;
            }

            min_op_type = std::min(min_op_type, op->type);
            // [v68] baked F16 weights make libQnnHtpPrepare emit ConvLayer.fp16
            // (a v69+ only op) which fails DSP-side activation. Weights must stay
            // dynamic graph inputs so Prepare emits the v68-native MatMul op.
            // Bake is opt-in via QNN_ENABLE_WEIGHT_BAKE=1 (dead on v68, kept for
            // newer HTP archs).
            if (device == QNN_BACKEND_NPU && getenv("QNN_ENABLE_WEIGHT_BAKE") && op->op == GGML_OP_MUL_MAT &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F16 && op->src[0]->data) {
                output += "#w:";
                output += op->src[0]->name[0] ? op->src[0]->name : std::to_string((uintptr_t) op->src[0]);
            }
            if (is_start) {
                qnn::get_qnn_op_desc(op, is_start, override_type, output);
                is_start = false;
            } else {
                output += '#';
                qnn::get_qnn_op_desc(op, is_start, override_type, output);
            }
        }
    }

    if (cgraph->n_nodes > 1) {
        auto * last_op = cgraph->nodes[cgraph->n_nodes - 1];
        output += qnn::get_ggml_type_name(last_op->type);
        output += '_';
        qnn::append_tensor_shape_and_type(last_op, output);
    }

    return min_op_type;
}

qnn_graph::qnn_graph(const std::string & graph_name, backend_index_type device, qnn_instance_ptr qnn_instance,
                     htp_precision precision, size_t vtcm_size_in_mb) :
    _graph_name(graph_name),
    _device(device),
    _qnn_instance(qnn_instance) {
    QNN_LOG_DEBUG("[%s][%s]creating\n", get_backend_name(device), graph_name.c_str());

    auto              qnn_interface = qnn_instance->get_qnn_interface();
    auto              qnn_context   = qnn_instance->get_qnn_context_handle();
    Qnn_ErrorHandle_t error         = QNN_SUCCESS;
    Qnn_GraphHandle_t graph_handle  = nullptr;
    if (device == QNN_BACKEND_NPU) {
        // TODO: fix graph config here for NPU
        // [v68] each config can steer libQnnHtpPrepare into weight-packing /
        // HVX / VTCM-resident graph paths whose ops are missing in the v68 skel.
        // Env toggles allow A/B testing which config forces the bad op choice.
        std::vector<const QnnGraph_Config_t *> graph_configs;

        if (!getenv("QNN_HTP_DISABLE_HVX")) {
            auto hvx_config = make_graph_config(&kDefaultHvxConfig);
            graph_configs.push_back(&hvx_config);
        }

        auto dlbc_config = make_graph_config(&kDefaultDlbcConfig);
        graph_configs.push_back(&dlbc_config);

        auto opt_config = make_graph_config(&kDefaultOptConfig);
        graph_configs.push_back(&opt_config);

        auto weights_packing_config = make_graph_config(&kDisableWeightsPackingConfig);
        graph_configs.push_back(&weights_packing_config);

        if (!getenv("QNN_HTP_DISABLE_VTCM")) {
            auto vctm_sub_config = make_vtcm_config(vtcm_size_in_mb);
            auto vtcm_config     = make_graph_config(&vctm_sub_config);
            graph_configs.push_back(&vtcm_config);
        }

        if (precision == qnn_graph::kHtpFp16 && !getenv("QNN_HTP_DISABLE_PRECISION")) {
            auto precision_config = make_graph_config(&kHtpPrecisionConfigF16);
            graph_configs.push_back(&precision_config);
            QNN_LOG_DEBUG("[%s][%s]set precision to F16\n", get_backend_name(device), graph_name.c_str());
        }

        graph_configs.push_back(nullptr);
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), graph_configs.data(), &graph_handle);
    } else {
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), nullptr, &graph_handle);
    }

    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s\n", get_backend_name(device), graph_name.c_str(),
                      get_qnn_error_string(error));
        return;
    }

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
    if (device == QNN_BACKEND_NPU) {
        _event_tracer = std::make_shared<qnn_event_tracer>(
            graph_name, qnn_interface, qnn_instance->get_qnn_backend_handle(), qnn_event_tracer::PROFILE_OP_TRACE);
    }
#endif

    _graph_handle  = graph_handle;
    _qnn_interface = qnn_interface;
    QNN_LOG_DEBUG("[%s][%s]create succeed\n", get_backend_name(device), graph_name.c_str());
}

qnn_graph::~qnn_graph() {
    QNN_LOG_DEBUG("[%s][%s]destroy\n", get_backend_name(_device), _graph_name.c_str());
}

bool qnn_graph::build_graph_from_ggml_graph(const ggml_cgraph * cgraph) {
    QNN_LOG_DEBUG("[%s][%s]build start\n", get_backend_name(_device), _graph_name.c_str());

    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
    int                 rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s][%s]rank: %d, graph_nodes: %d, input_set: %d, output_set: %d\n", get_backend_name(_device),
                  _graph_name.c_str(), rank, cgraph->n_nodes, int(inputs.size()), int(outputs.size()));

    {
        static_assert(
            GGML_TYPE_COUNT > GGML_TYPE_Q8_0 && GGML_TYPE_Q8_0 > GGML_TYPE_F16 && GGML_TYPE_F16 > GGML_TYPE_F32,
            "GGML_TYPE enum order is not correct");

        SCOPED_PERFORMANCE_TRACKER("[%s][%s]build_graph_from_ggml_graph", get_backend_name(_device),
                                   _graph_name.c_str());

        auto override_data_type = get_override_data_type(inputs, outputs, _device);
        _io_data_type           = override_data_type;
        if (override_data_type != GGML_TYPE_COUNT) {
            QNN_LOG_DEBUG("[%s][%s]set override_data_type: %s\n", get_backend_name(_device), _graph_name.c_str(),
                          ggml_type_name(override_data_type));
        }

        qnn_tensor_cache_t tensor_cache;

        // [v68] F16 matmul weights are baked into the graph as QNN STATIC tensors
        // only when QNN_ENABLE_WEIGHT_BAKE=1: on v68 Prepare translates
        // static-weight matmuls into ConvLayer.fp16 (v69+ only op) which fails
        // DSP-side activation, so weights stay dynamic inputs by default.
        std::unordered_set<const ggml_tensor *> weight_tensors;
        if (_device == QNN_BACKEND_NPU && getenv("QNN_ENABLE_WEIGHT_BAKE")) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                auto * dst = cgraph->nodes[i];
                if (dst->op == GGML_OP_MUL_MAT && dst->src[0] && dst->src[0]->type == GGML_TYPE_F16 &&
                    dst->src[0]->data) {
                    weight_tensors.insert(dst->src[0]);
                }
            }
        }
        _baked_weights = weight_tensors;

        qnn_tensor_array_t input_tensors;
        for (auto * tensor : inputs) {
            if (weight_tensors.count(tensor)) {
                auto weight_tensor = create_tensor_with_cache(tensor, ggml_qnn_tensor::PARAMETER, rank,
                                                              override_data_type, _device, _graph_handle,
                                                              _qnn_instance, tensor_cache);
                weight_tensor->set_data_buffer(reinterpret_cast<const uint8_t *>(tensor->data),
                                               ggml_nbytes(tensor));
                QNN_LOG_DEBUG("[%s][%s]baked weight(%s) as static tensor, size: %d\n", get_backend_name(_device),
                              _graph_name.c_str(), tensor->name, (int) ggml_nbytes(tensor));
                // static tensors are serialized at finalize; not runtime inputs
                continue;
            }
            input_tensors.push_back(create_tensor_with_cache(tensor, ggml_qnn_tensor::INPUT, rank, override_data_type,
                                                             _device, _graph_handle, _qnn_instance, tensor_cache));
        }
        // [v68] graph outputs must keep the overridden type too, otherwise the
        // matmul op config inserts an output Cast node (missing in the v68 skel)
        auto output_override = _device == QNN_BACKEND_NPU ? override_data_type : ggml_type(GGML_TYPE_COUNT);
        auto output_tensors = create_tensors_with_cache(outputs, ggml_qnn_tensor::OUTPUT, rank, output_override,
                                                        _device, _graph_handle, _qnn_instance, tensor_cache);
        qnn_op_config_array_t operations;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * dst = cgraph->nodes[i];
            if (ggml_is_empty(dst)) {
                continue;
            }

            if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW || dst->op == GGML_OP_PERMUTE) {
                // TODO: remove GGML_OP_VIEW after view op is supported
                continue;
            }

#ifndef NDEBUG
            {
                std::string op_desc;
                get_qnn_op_desc(dst, true, GGML_TYPE_COUNT, op_desc);
                QNN_LOG_DEBUG("[%s]create op(%s) with qnn op(%s)\n", get_backend_name(_device), op_desc.c_str(),
                              get_qnn_op_name(dst));
            }
#endif
            auto operation = create_operation_from_op_tensor(dst, dst->name, rank, _device, _graph_handle,
                                                             _qnn_instance, tensor_cache);  // TODO: fix op name
            operations.push_back(operation);
        }

        _tensor_inputs  = std::move(input_tensors);
        _tensor_outputs = std::move(output_tensors);
        _operations     = std::move(operations);
        if (!finalize()) {
            return false;
        }
    }

    QNN_LOG_DEBUG("[%s][%s]build succeed\n", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::execute(const ggml_cgraph * cgraph, std::shared_ptr<qnn_convert_context_t> convert_context) {
    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
    {
        SCOPED_PERFORMANCE_TRACKER("[%s][%s]get_io_tensors_from_graph", get_backend_name(_device), _graph_name.c_str());
#ifdef NDEBUG
        get_io_tensors_from_graph(cgraph, inputs, outputs);
#else
        int rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
        QNN_LOG_DEBUG("[%s]rank: %d, input_set: %d, output_set: %d\n", get_backend_name(_device), rank,
                      int(inputs.size()), int(outputs.size()));
#endif
    }

    // graph output staging: non-empty when the graph output type differs from the
    // ggml tensor type (NPU F16 graphs); filled right after the execute block below
    std::vector<qnn::qnn_buffer_ptr> output_staging_buffers;
    ggml_type                       output_staging_type = GGML_TYPE_COUNT;

    // [v68] baked weights are already serialized in the graph, drop them from the
    // runtime input set so bindings stay aligned with the registered graph inputs
    if (!_baked_weights.empty()) {
        qnn::ggml_tensor_array_t live_inputs;
        for (auto * tensor : inputs) {
            if (!_baked_weights.count(tensor)) {
                live_inputs.push_back(tensor);
            }
        }
        inputs = std::move(live_inputs);
    }

    {
        SCOPED_PERFORMANCE_TRACKER("[%s][%s]bind_tensors", get_backend_name(_device), _graph_name.c_str());
        // reuse the override type recorded at build time (recomputing it here would
        // ignore the baked weights and fall back to F32)
        auto override_data_type = _io_data_type;
        if (override_data_type != GGML_TYPE_COUNT) {
            QNN_LOG_DEBUG("[%s][%s]override_data_type: %s\n", get_backend_name(_device), _graph_name.c_str(),
                          ggml_type_name(override_data_type));
            auto buffers = convert(convert_context, inputs, override_data_type);
            if (!qnn::bind_tensors_with_custom_buffers(inputs, buffers, _tensor_inputs, _qnn_tensor_inputs)) {
                QNN_LOG_ERROR("[%s][%s]bind input tensors failed\n", get_backend_name(_device), _graph_name.c_str());
                return false;
            }
        } else {
            if (!qnn::bind_tensors(inputs, _tensor_inputs, _qnn_tensor_inputs)) {
                QNN_LOG_ERROR("[%s][%s]bind input tensors failed\n", get_backend_name(_device), _graph_name.c_str());
                return false;
            }
        }

        const bool need_output_deconvert = override_data_type != GGML_TYPE_COUNT;
        if (need_output_deconvert) {
            output_staging_type = override_data_type;
            // bind outputs to host staging buffers of the graph type; unbind() will
            // copy the DSP results back into them, and we convert to F32 afterwards
            output_staging_buffers.resize(outputs.size());
            _convert_output_buffers.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto * out  = outputs[i];
                const size_t nels = ggml_nelements(out);
                const size_t nbytes = nels * ggml_type_size(override_data_type);
                auto &       buf   = _convert_output_buffers[i];
                if (!buf || buf->get_size() < nbytes) {
                    buf = std::make_shared<qnn::qnn_mem_buffer>(nbytes);
                }
                output_staging_buffers[i] = buf;
            }
            if (!qnn::bind_tensors_with_custom_buffers(outputs, output_staging_buffers, _tensor_outputs,
                                                       _qnn_tensor_outputs)) {
                QNN_LOG_ERROR("[%s][%s]bind output tensors failed\n", get_backend_name(_device), _graph_name.c_str());
                return false;
            }
        } else {
            if (!qnn::bind_tensors(outputs, _tensor_outputs, _qnn_tensor_outputs)) {
                QNN_LOG_ERROR("[%s][%s]bind output tensors failed\n", get_backend_name(_device), _graph_name.c_str());
                return false;
            }
        }
    }

    {
        SCOPED_PERFORMANCE_TRACKER("[%s][%s]execute", get_backend_name(_device), _graph_name.c_str());
        auto & qnn_tensor_inputs  = _qnn_tensor_inputs;
        auto & qnn_tensor_outputs = _qnn_tensor_outputs;
        auto   error              = _qnn_interface->qnn_graph_execute(_graph_handle, qnn_tensor_inputs.data(),
                                                                      qnn_tensor_inputs.size(), qnn_tensor_outputs.data(),
                                                                      qnn_tensor_outputs.size(), GRAPH_PROFILE_HANDLE, nullptr);
        unbind_tensors(_tensor_inputs);
        unbind_tensors(_tensor_outputs);
        if (error != QNN_SUCCESS) {
            if (_device == QNN_BACKEND_NPU && error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION) {
                QNN_LOG_WARN("[%s][%s][execute]NPU crashed. SSR detected. Caused QNN graph execute error.\n",
                             get_backend_name(_device), _graph_name.c_str());
            } else {
                QNN_LOG_ERROR("[%s][%s][execute]error: %s\n", get_backend_name(_device), _graph_name.c_str(),
                              get_qnn_error_string(error));
            }
            return false;
        }

        if (!output_staging_buffers.empty()) {
            // unbind() above already copied the DSP results back into the staging
            // buffers; convert them from the graph type into the ggml tensors
            const auto to_float = ggml_get_type_traits(output_staging_type)->to_float;
            GGML_ASSERT(to_float);
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto *     out  = outputs[i];
                const auto nels = ggml_nelements(out);
                to_float((const char *) output_staging_buffers[i]->get_buffer(), (float *) out->data, (int) nels);
            }
        }

        QNN_LOG_DEBUG("[%s][%s]execute succeed\n", get_backend_name(_device), _graph_name.c_str());
    }

    GRAPH_PROFILE_PRINT();
    return true;
}

bool qnn_graph::finalize() {
    SCOPED_PERFORMANCE_TRACKER("[%s][%s]finalize", get_backend_name(_device), _graph_name.c_str());

    if (!qnn::add_op_to_graph(_graph_handle, _operations)) {
        QNN_LOG_ERROR("[%s]add nodes failed\n", _graph_name.c_str());
        return false;
    }

    auto error = _qnn_interface->qnn_graph_finalize(_graph_handle, GRAPH_PROFILE_HANDLE, nullptr);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]qnn_graph_finalize.error: %s\n", get_backend_name(_device), _graph_name.c_str(),
                      get_qnn_error_string(error));
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]finalize succeed\n", get_backend_name(_device), _graph_name.c_str());
    return true;
}

}  // namespace qnn
